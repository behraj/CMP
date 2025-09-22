import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from clip import clip
import torch.nn.functional as F
import random
from .coop import load_clip_to_cpu
from tqdm import tqdm
from copy import deepcopy

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from trainers.classification.base_learner import VLBaseLearner
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tools.ood_search import ClassNameIterator

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames_id, classnames_ood, clip_model):
        super().__init__()
        n_cls = len(classnames_id)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)]) # (depth- 1) * (n_ctx, 512)

        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1) # (depth- 1) * (512, 768)


        classnames_id = [name.replace("_", " ") for name in classnames_id]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames_id]
        prompts = [prompt_prefix + " " + name + "." for name in classnames_id]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.prompt_prefix = prompt_prefix
        self.classnames_ood = classnames_ood
        self.dtype = dtype
        self.n_ctx = n_ctx
        self.cfg = cfg
        self.token_embedding = clip_model.token_embedding
        self.temp_ood = cfg.OOD.TEMP
        self.classname_gen = ClassNameIterator(classnames_ood, n_cls, cfg.OOD.SAMPLING)

    def construct_prompts(self, ctx, prefix, suffix, label=None):

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):

        # image prompt
        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))


        # 1. id text prompt using tuned prompt
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts_id = self.construct_prompts(ctx, prefix, suffix)

        # get ood classnames
        if self.cfg.OOD.LOSS_REG:
            # if len(self.classnames_ood) > prompts_id.shape[0]:
            #     classnames_ood = random.sample(self.classnames_ood, self.n_cls)
            # else:
            #     classnames_ood = self.classnames_ood
            classnames_ood = self.classname_gen.next_batch()
                
            classnames_ood = self.classname_gen.next_batch()
        
            classnames_ood = [name.replace("_", " ") for name in classnames_ood]

            # 2. ood prompt using tuned ctx
            prompts_ood = [self.prompt_prefix + " " + name + "." for name in classnames_ood]
            prompts_ood_ft_tokenized = torch.cat([clip.tokenize(p) for p in prompts_ood]).cuda()

            with torch.no_grad():
                embedding_ood = self.token_embedding(prompts_ood_ft_tokenized).type(self.dtype)

            token_prefix_ood = embedding_ood[:, :1, :]
            token_suffix_ood = embedding_ood[:, 1 + self.n_ctx:, :]

            ctx_ood = self.ctx
            if ctx_ood.dim() == 2:
                ctx_ood = ctx_ood.unsqueeze(0).expand(len(classnames_ood), -1, -1)

            prompts_ood_ft = self.construct_prompts(ctx_ood, token_prefix_ood, token_suffix_ood)

            # 3. ood prompt using fixed temp "a photo of a"
            prompts_ood_zs = [self.temp_ood + " " + name + "." for name in classnames_ood]
            prompts_ood_zs_tokenized = torch.cat([clip.tokenize(p) for p in prompts_ood_zs]).cuda()

            return prompts_id, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts, prompts_ood_zs_tokenized, prompts_ood_ft, prompts_ood_ft_tokenized   # pass here original, as for visual 768 is required

        else:
            # Now the other way around
            # We will project the textual prompts from 512 to 768
            return prompts_id, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames_id, classnames_ood, clip_model, clip_zs):
        super().__init__()
        self.classnames_id = classnames_id
        self.classnames_ood = classnames_ood
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames_id, classnames_ood, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.cfg = cfg
        self.alpha = self.cfg.OOD.ALPHA

        # load zero-shot CLIP
        clip_zs = clip_zs.cuda()
        self.encode_text = clip_zs.encode_text


    def forward(self, image, label=None):

        # get text features
        if self.cfg.OOD.LOSS_REG:
            prompts_id, shared_ctx, deep_compound_prompts_text_id, deep_compound_prompts_vision,\
                prompts_ood_zs_tokenized, prompts_ood_ft, prompts_ood_ft_tokenized = self.prompt_learner()
            prompts_id_tokenized = self.tokenized_prompts.cuda()

            prompts_mix = torch.cat([prompts_id, prompts_ood_ft], dim=0)
            tokenized_prompts_mix = torch.cat([prompts_id_tokenized, prompts_ood_ft_tokenized], dim=0)
            text_features_mix = self.text_encoder(prompts_mix, tokenized_prompts_mix, deep_compound_prompts_text_id)

            text_features_mix = text_features_mix / text_features_mix.norm(dim=-1, keepdim=True)
            text_features_id = text_features_mix[:prompts_id.size(0)]
            text_features_ood_ft = text_features_mix[prompts_id.size(0):]

            with torch.no_grad():
                text_features_ood_zs = self.encode_text(prompts_ood_zs_tokenized).type(self.dtype)
                text_features_ood_zs = text_features_ood_zs / text_features_ood_zs.norm(dim=-1, keepdim=True) 
        else:
            prompts_id_tokenized = self.tokenized_prompts
            prompts_id, shared_ctx, deep_compound_prompts_text_id, deep_compound_prompts_vision = self.prompt_learner()
            text_features_id = self.text_encoder(prompts_id, prompts_id_tokenized, deep_compound_prompts_text_id)
            text_features_id = text_features_id / text_features_id.norm(dim=-1, keepdim=True)


        # get image features
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features_id.t()

        # build loss
        if self.prompt_learner.training:
            loss_ori = F.cross_entropy(logits, label)

            if self.cfg.OOD.LOSS_REG:
                cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
                score = cos(text_features_ood_ft, text_features_ood_zs)
                loss_text_reg = 1.0-torch.mean(score)
                loss = loss_ori + self.alpha * loss_text_reg
            else:
                loss =  loss_ori

            return loss

        return logits, image_features, text_features_id


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe(VLBaseLearner):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames_id = self.dm.dataset.classnames

        ######## ood #########
        if self.cfg.OOD.LOSS_REG:
            classnames_ood = self.ood_classnames
            length = len(classnames_id)
        else:
            classnames_ood = None
        
        clip_zs = self.load_zero_shot_clip_to_cpu()

        ######## ood #########

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames_id, classnames_ood, clip_model, clip_zs)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # self.model = nn.DataParallel(self.model)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
