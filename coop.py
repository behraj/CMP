import os.path as osp
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from clip import clip
import torch.nn.functional as F

from tqdm import tqdm
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from trainers.classification.base_learner import VLBaseLearner
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tools.ood_search import ClassNameIterator

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    #"EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

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
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
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

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames_id, classnames_ood, clip_model):
        super().__init__()
        n_cls = len(classnames_id)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames_id = [name.replace("_", " ") for name in classnames_id]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames_id]
        prompts = [prompt_prefix + " " + name + "." for name in classnames_id]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.prompt_prefix = prompt_prefix
        self.classnames_ood = classnames_ood
        self.dtype = dtype
        self.n_ctx = n_ctx
        self.cfg = cfg
        self.token_embedding = clip_model.token_embedding
        self.temp_ood = cfg.OOD.TEMP
        self.classname_gen = ClassNameIterator(classnames_ood, n_cls, cfg.OOD.SAMPLING)


    def forward(self):

        # 1. id prompt using tuned prompt
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts_id = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        else:
            raise ValueError
        
        # get ood classnames
        if self.cfg.OOD.LOSS_REG:
            # if len(self.classnames_ood) > prompts_id.shape[0]:
            #     classnames_ood = random.sample(self.classnames_ood, self.n_cls)
            # else:
            #     classnames_ood = self.classnames_ood

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

            if self.class_token_position == "end":
                prompts_ood_ft = torch.cat(
                    [
                        token_prefix_ood,  # (n_cls, 1, dim)
                        ctx_ood,     # (n_cls, n_ctx, dim)
                        token_suffix_ood,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                raise ValueError
            
            # 3. ood prompt using fixed temp "a photo of a"
            prompts_ood_zs = [self.temp_ood + " " + name + "." for name in classnames_ood]
            prompts_ood_zs_tokenized = torch.cat([clip.tokenize(p) for p in prompts_ood_zs]).cuda()

            return prompts_id, prompts_ood_zs_tokenized, prompts_ood_ft, prompts_ood_ft_tokenized
        
        else:
            return prompts_id


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames_id, classnames_ood, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames_id = classnames_id
        self.classnames_ood = classnames_ood
        self.prompt_learner = PromptLearner(cfg, classnames_id, classnames_ood, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.alpha = self.cfg.OOD.ALPHA
        self.encode_text = clip_model.encode_text

    def forward(self, image, label=None):

        #image features
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)


        # text features
        if self.cfg.OOD.LOSS_REG:
            prompts_id, prompts_ood_zs_tokenized, prompts_ood_ft, prompts_ood_ft_tokenized = self.prompt_learner()
            prompts_id_tokenized = self.tokenized_prompts.cuda()

            prompts_mix = torch.cat([prompts_id, prompts_ood_ft], dim=0)
            tokenized_prompts_mix = torch.cat([prompts_id_tokenized, prompts_ood_ft_tokenized], dim=0)
            text_features_mix = self.text_encoder(prompts_mix, tokenized_prompts_mix)

            text_features_mix = text_features_mix / text_features_mix.norm(dim=-1, keepdim=True)

            text_features_id = text_features_mix[:prompts_id.size(0)]
            text_features_ood_ft = text_features_mix[prompts_id.size(0):]

            with torch.no_grad():
                text_features_ood_zs = self.encode_text(prompts_ood_zs_tokenized).type(self.dtype)
                text_features_ood_zs = text_features_ood_zs / text_features_ood_zs.norm(dim=-1, keepdim=True) 
        else:
            prompts_id = self.prompt_learner()
            prompts_id_tokenized = self.tokenized_prompts
            text_features_id = self.text_encoder(prompts_id, prompts_id_tokenized)
            text_features_id = text_features_id / text_features_id.norm(dim=-1, keepdim=True)

        # get logits
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


@TRAINER_REGISTRY.register()
class CoOp(VLBaseLearner):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames_id = self.dm.dataset.classnames

        ######## ood #########
        if self.cfg.OOD.LOSS_REG:
            classnames_ood = self.ood_classnames
            length = len(classnames_id)
        else:
            classnames_ood = None
        ######## ood #########

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames_id, classnames_ood, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

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

        prec = self.cfg.TRAINER.COOP.PREC
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

        loss_summary = {
            "loss": loss.item()
        }

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
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
