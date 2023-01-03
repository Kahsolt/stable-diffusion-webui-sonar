import os
from PIL import Image
from typing import List
from pprint import pprint as pp

import gradio as gr
import torch
from torch import Tensor
import numpy as np
from tqdm.auto import trange

import modules.scripts as scripts
from modules.shared import state
from modules.processing import *
from modules.sd_samplers import *
from modules.prompt_parser import ScheduledPromptConditioning, MulticondLearnedConditioning
from k_diffusion.sampling import to_d, get_ancestral_step
from ldm.models.diffusion.ddpm import LatentDiffusion

DEFAULT_SAMPLER            = 'Euler a'
DEFAULT_MOMENTUM           = 0.95
DEFAULT_MOMENTUM_HIST      = 0.75
DEFAULT_MOMENTUM_HIST_INIT = 'zero'
DEFAULT_MOMENTUM_SIGN      = 'pos'
DEFAULT_GRAD_C_ITER        = 0
DEFAULT_GRAD_C_ALPHA       = -0.002
DEFAULT_GRAD_C_SKIP        = 5
DEFAULT_GRAD_X_ITER        = 0
DEFAULT_GRAD_X_ALPHA       = -0.02
DEFAULT_GRAD_FUZZY         = False
DEFAULT_REF_METH           = 'linear'
DEFAULT_REF_HGF            = 0.01
DEFAULT_REF_MIN_STEP       = 0.0
DEFAULT_REF_MAX_STEP       = 0.75
DEFAULT_REF_IMG            = None

CHOICE_MOMENTUM_SIGN      = ['pos', 'neg', 'rand']
CHOICE_MOMENTUM_HIST_INIT = ['zero', 'rand_init', 'rand_new']
CHOICE_REF_METH           = ['linear', 'euler']

# debug save latent featmap (when `Euler a`)
#FEAT_MAP_PATH = 'C:\sd-webui_featmaps'
FEAT_MAP_PATH = None

# the current setting (the wrappers are too deep, we pass it by global var)
settings = {
    'sampler':            DEFAULT_SAMPLER,

    'momentum':           DEFAULT_MOMENTUM,
    'momentum_hist':      DEFAULT_MOMENTUM_HIST,
    'momentum_hist_init': DEFAULT_MOMENTUM_HIST_INIT,
    'momentum_sign':      DEFAULT_MOMENTUM_SIGN,

    'grad_c_iter':        DEFAULT_GRAD_C_ITER,
    'grad_c_alpha':       DEFAULT_GRAD_C_ALPHA,
    'grad_c_skip':        DEFAULT_GRAD_C_SKIP,
    'grad_x_iter':        DEFAULT_GRAD_X_ITER,
    'grad_x_alpha':       DEFAULT_GRAD_X_ALPHA,
    'grad_fuzzy':         DEFAULT_GRAD_FUZZY,

    'ref_meth':           DEFAULT_REF_METH,
    'ref_hgf':            DEFAULT_REF_HGF,
    'ref_min_step':       DEFAULT_REF_MIN_STEP,
    'ref_max_step':       DEFAULT_REF_MAX_STEP,
    'ref_img':            DEFAULT_REF_IMG,
}

# ↓↓↓ the following is modified from 'modules/processing.py' ↓↓↓

def process_images(p: StableDiffusionProcessing) -> Processed:
    stored_opts = {k: opts.data[k] for k in p.override_settings.keys()}

    try:
        for k, v in p.override_settings.items():
            setattr(opts, k, v)
            if k == 'sd_hypernetwork': shared.reload_hypernetworks()  # make onchange call for changing hypernet
            if k == 'sd_model_checkpoint': sd_models.reload_model_weights()  # make onchange call for changing SD model
            if k == 'sd_vae': sd_vae.reload_vae_weights()  # make onchange call for changing VAE

        res = process_images_inner(p)

    finally:
        # restore opts to original state
        if p.override_settings_restore_afterwards:
            for k, v in stored_opts.items():
                setattr(opts, k, v)
                if k == 'sd_hypernetwork': shared.reload_hypernetworks()
                if k == 'sd_model_checkpoint': sd_models.reload_model_weights()
                if k == 'sd_vae': sd_vae.reload_vae_weights()

    return res

def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    modules.sd_hijack.model_hijack.clear_comments()

    comments = {}

    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
    else:
        p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    with open(os.path.join(shared.script_path, "params.txt"), "w", encoding="utf8") as file:
        processed = Processed(p, [], p.seed, "")
        file.write(processed.infotext(p, 0))

    if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
        model_hijack.embedding_db.load_textual_inversion_embeddings()

    if p.scripts is not None:
        p.scripts.process(p)

    infotexts = []
    output_images = []

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        if state.job_count == -1:
            state.job_count = p.n_iter

        for n in range(p.n_iter):
            if state.skipped:
                state.skipped = False
            
            if state.interrupted:
                break

            prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if len(prompts) == 0:
                break

            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

            with devices.autocast():
                uc = prompt_parser.get_learned_conditioning(shared.sd_model, negative_prompts, p.steps)
                c = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, prompts, p.steps)

            if len(model_hijack.comments) > 0:
                for comment in model_hijack.comments:
                    comments[comment] = 1

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            with devices.autocast():
                if   isinstance(p, StableDiffusionProcessingTxt2Img): sample_func = StableDiffusionProcessingTxt2Img_sample
                elif isinstance(p, StableDiffusionProcessingImg2Img): sample_func = StableDiffusionProcessingImg2Img_sample
                else: raise ValueError
                samples_ddim = sample_func(p, conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)
            
            x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            del samples_ddim

            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                lowvram.send_everything_to_cpu()

            devices.torch_gc()

            if opts.filter_nsfw:
                import modules.safety as safety
                x_samples_ddim = modules.safety.censor_batch(x_samples_ddim)

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                if opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                text = infotext(n, i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)

            del x_samples_ddim 

            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext()
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    devices.torch_gc()

    res = Processed(p, output_images, p.all_seeds[0], infotext() + "".join(["\n\n" + x for x in comments]), subseed=p.all_subseeds[0], index_of_first_image=index_of_first_image, infotexts=infotexts)

    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res

def StableDiffusionProcessingTxt2Img_sample(self:StableDiffusionProcessingTxt2Img, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
    # hijack the sampler~
    self.sampler = create_sampler(self.sd_model)

    latent_scale_mode = shared.latent_upscale_modes.get(self.hr_upscaler, None) if self.hr_upscaler is not None else shared.latent_upscale_default_mode
    if self.enable_hr and latent_scale_mode is None:
        assert len([x for x in shared.sd_upscalers if x.name == self.hr_upscaler]) > 0, f"could not find upscaler named {self.hr_upscaler}"

    x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
    samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))

    if not self.enable_hr:
        return samples

    target_width = int(self.width * self.hr_scale)
    target_height = int(self.height * self.hr_scale)

    def save_intermediate(image, index):
        """saves image before applying hires fix, if enabled in options; takes as an argument either an image or batch with latent space images"""

        if not opts.save or self.do_not_save_samples or not opts.save_images_before_highres_fix:
            return

        if not isinstance(image, Image.Image):
            image = sd_samplers.sample_to_image(image, index)

        images.save_image(image, self.outpath_samples, "", seeds[index], prompts[index], opts.samples_format, suffix="-before-highres-fix")

    if latent_scale_mode is not None:
        for i in range(samples.shape[0]):
            save_intermediate(samples, i)

        samples = torch.nn.functional.interpolate(samples, size=(target_height // opt_f, target_width // opt_f), mode=latent_scale_mode)

        # Avoid making the inpainting conditioning unless necessary as
        # this does need some extra compute to decode / encode the image again.
        if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
            image_conditioning = self.img2img_image_conditioning(decode_first_stage(self.sd_model, samples), samples)
        else:
            image_conditioning = self.txt2img_image_conditioning(samples)
    else:
        decoded_samples = decode_first_stage(self.sd_model, samples)
        lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

        batch_images = []
        for i, x_sample in enumerate(lowres_samples):
            x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
            x_sample = x_sample.astype(np.uint8)
            image = Image.fromarray(x_sample)

            save_intermediate(image, i)

            image = images.resize_image(0, image, target_width, target_height, upscaler_name=self.hr_upscaler)
            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            batch_images.append(image)

        decoded_samples = torch.from_numpy(np.array(batch_images))
        decoded_samples = decoded_samples.to(shared.device)
        decoded_samples = 2. * decoded_samples - 1.

        samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))

        image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)

    shared.state.nextjob()

    noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, p=self)

    # GC now before running the next img2img to prevent running out of memory
    x = None
    devices.torch_gc()

    # hijack the sampler~
    self.sampler = create_sampler(self.sd_model)
    samples = self.sampler.sample_img2img(self, samples, noise, conditioning, unconditional_conditioning, steps=self.steps, image_conditioning=image_conditioning)

    return samples

def StableDiffusionProcessingImg2Img_sample(self:StableDiffusionProcessingImg2Img, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
    x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)

    if self.initial_noise_multiplier != 1.0:
        self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
        x *= self.initial_noise_multiplier

    # hijack the sampler~
    self.sampler = create_sampler(self.sd_model)
    samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)

    if self.mask is not None:
        samples = samples * self.nmask + self.init_latent * self.mask

    del x
    devices.torch_gc()

    return samples

# ↑↑↑ the above is modified from 'modules/processing.py' ↑↑↑


# ↓↓↓ the following is modified from 'k_diffusion/sampling.py' ↓↓↓

def show_featmap(x, title=''):
    if not FEAT_MAP_PATH: return

    import os
    import matplotlib.pyplot as plt
    os.makedirs(FEAT_MAP_PATH, exist_ok=True)

    x_np = x[0].cpu().numpy()    # [C=4, H=64, W=64]
    x_np_abs = np.abs(x_np)
    print(f'[{title}]')
    print('   x_np:',     x_np    .max(), x_np    .min(), x_np    .mean(), x_np    .std())
    print('   x_np_abs:', x_np_abs.max(), x_np_abs.min(), x_np_abs.mean(), x_np_abs.std())
    for i in range(4):
        plt.axis('off')
        plt.subplot(2, 2, i+1)
        plt.imshow(x_np[i])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FEAT_MAP_PATH, f'{title}.png'))

@torch.no_grad()
def sample_naive(model:CFGDenoiser, x:Tensor, sigmas:List, extra_args={}, callback=None, *args):
    '''
    采样的本质是寻找降噪模型在某输入维度(图隐层维度)上的不动点，通过反复查询、依近似梯度来优化选点
    你可以基于该模板快速开发 K-Diffusion 采样器 :)
    '''
    # type(model)                                                               modules.sd_samplers.CFGDenoiser
    # type(model.inner_model)                                                   k_diffusion.external.CompVisDenoiser
    # type(model.inner_model.inner_model)                                       ldm.models.diffusion.ddpm.LatentDiffusion
    # type(model.inner_model.inner_model.first_stage_model)                     ldm.models.autoencoder.AutoencoderKL
    # type(model.inner_model.inner_model.cond_stage_model)                      modules.sd_hijack.FrozenCLIPEmbedderWithCustomWords
    # type(model.inner_model.inner_model.cond_stage_model.wrapped)              ldm.modules.encoders.modules.FrozenCLIPEmbedder
    # type(model.inner_model.inner_model.cond_stage_model.wrapped.tokenizer)    transformers.models.clip.tokenization_clip.CLIPTokenizer
    # type(model.inner_model.inner_model.cond_stage_model.wrapped.transformer)  transformers.models.clip.modeling_clip.CLIPTextModel
    # type(model.inner_model.inner_model.model)                                 ldm.models.diffusion.ddpm.DiffusionWrapper
    # type(model.inner_model.inner_model.model.diffusion_model)                 ldm.modules.diffusionmodules.openaimodel.UNetModel
    # x                                                                         Tensor([B, C=4, H, W]), x8 downsampled
    # sigmas                                                                    Tensor([T]), steps
    # extra_args['cond']                                                        MulticondLearnedConditioning, prompt cond
    # extra_args['uncond']                                                      List[List[ScheduledPromptConditioning]], negaivte prompt cond
    # extra_args['image_cond']                                                  Tensor(), mask for img2img; Tensor([1, 5, 1, 1]), dummy for txt2img
    # extra_args['cond_scale']                                                  int, e.g.: 7.0
    # callback                                                                  KDiffusionSampler.callback_state(dict)
    # *args                                                                     see `sampler_extra_params_sonar`

    s_in = x.new_ones([x.shape[0]])         # expand_dim
    for i in trange(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'i': i, 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)    # dy/dx
        dt = sigmas[i + 1] - sigmas[i]      # dx
        x = x + d * dt                      # x += dx * dy/dx
    return x

@torch.no_grad()
def sample_naive_ex(model:CFGDenoiser, x:Tensor, sigmas:List, extra_args={}, callback=None):
    sd_model: LatentDiffusion = model.inner_model.inner_model

    momentum           = settings['momentum']
    momentum_sign      = settings['momentum_sign']
    momentum_hist      = settings['momentum_hist']
    momentum_hist_init = settings['momentum_hist_init']
    grad_c_iter        = settings['grad_c_iter']
    grad_c_alpha       = settings['grad_c_alpha']
    grad_c_skip        = settings['grad_c_skip']
    grad_x_iter        = settings['grad_x_iter']
    grad_x_alpha       = settings['grad_x_alpha']
    grad_fuzzy         = settings['grad_fuzzy']
    ref_hgf            = settings['ref_hgf']
    ref_meth           = settings['ref_meth']
    ref_img            = settings['ref_img']
    ref_min_step       = settings['ref_min_step']
    ref_max_step       = settings['ref_max_step']

    # memorize delta momentum
    if   momentum_hist_init == 'zero':      history_d = 0
    elif momentum_hist_init == 'rand_init': history_d = x
    elif momentum_hist_init == 'rand_new':  history_d = torch.randn_like(x)
    else: raise ValueError(f'unknown momentum_hist_init: {momentum_hist_init}')

    # prepare ref_img latent
    if ref_img is not None:
        img = Image.open(ref_img).convert('RGB')
        x_ref = torch.from_numpy(np.asarray(img)).moveaxis(2, 0)    # [C=3, H, W]
        x_ref = (x_ref / 255) * 2 - 1
        x_ref = x_ref.unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)  # [B, C=3, H, W]
        x_ref = x_ref.to(sd_model.first_stage_model.device)

        with devices.autocast():
            latent_ref = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(x_ref))     # [B, C=4, H=64, W=64]

            avg_s = latent_ref.mean(dim=[2, 3], keepdim=True)
            std_s = latent_ref.std (dim=[2, 3], keepdim=True)
            ref_img_norm = (latent_ref - avg_s) / std_s
    
    # stochastics in gradient optimizing
    noise = None if grad_fuzzy else torch.randn_like(x)
    t     = None if grad_fuzzy else torch.randint(0, sd_model.num_timesteps, (x.shape[0],), device=sd_model.device).long()

    s_in = x.new_ones([x.shape[0]])
    n_steps = len(sigmas) - 1
    for i in trange(n_steps):
        if state.interrupted: break

        # denoise step
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'i': i, 'denoised': denoised})
        
        # grad step
        d = to_d(x, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigmas[i]

        # momentum step
        if momentum < 1.0:
            # decide correct direction
            sign = momentum_sign
            if sign == 'rand': sign = random.choice(['pos', 'neg'])

            # correct current `d` with momentum
            p = 1.0 - momentum
            if   sign == 'pos': momentum_d = (1.0 - p) * d + p * history_d
            elif sign == 'neg': momentum_d = (1.0 + p) * d - p * history_d
            else: raise ValueError(f'unknown momentum sign {sign}')
            
            # Euler method with momentum
            x = x + momentum_d * dt

            # update momentum history
            q = 1.0 - momentum_hist
            if (isinstance(history_d, int) and history_d == 0):
                history_d = momentum_d
            else:
                if   sign == 'pos': history_d = (1.0 - q) * history_d + q * momentum_d
                elif sign == 'neg': history_d = (1.0 + q) * history_d - q * momentum_d
                else: raise ValueError(f'unknown momentum sign {sign}')
        else:
            # Euler method original
            x = x + d * dt

        # guidance step
        if ref_img is not None and ref_hgf and ref_min_step <= i <= ref_max_step:
            # TODO: make scheduling for hgf?
            if ref_meth == 'euler':
                # rescale `ref_img` to match distribution
                avg_t = denoised.mean(dim=[1, 2, 3], keepdim=True)
                std_t = denoised.std (dim=[1, 2, 3], keepdim=True)
                ref_img_shift = ref_img_norm * std_t + avg_t

                d = to_d(x, sigmas[i], ref_img_shift)
                dt = (sigmas[i + 1] - sigmas[i]) * ref_hgf
                x = x + d * dt
            if ref_meth == 'linear':
                # rescale `ref_img` to match distribution
                avg_t = x.mean(dim=[1, 2, 3], keepdim=True)
                std_t = x.std (dim=[1, 2, 3], keepdim=True)
                ref_img_shift = ref_img_norm * std_t + avg_t

                x = (1 - ref_hgf) * x + ref_hgf * ref_img_shift

        # inprocess-optimizing prompt condition
        if i >= grad_c_skip and grad_c_iter:
            c = mlc_get_cond(extra_args['cond']).unsqueeze(dim=0)

            for i in trange(grad_c_iter):
                if state.interrupted: break

                with torch.enable_grad():
                    c = c.detach().clone() ; c.requires_grad = True
                    x = x.detach().clone() ; x.requires_grad = True

                    loss = get_latent_loss(sd_model, x, t, c, noise)
                    grad = torch.autograd.grad(loss, inputs=c, grad_outputs=loss)[0]

                with torch.no_grad():
                    print(f'loss_c: {loss.mean().item():.7f}, grad_c: {grad.mean().item():.7f}')

                c = c.detach() - grad.sign() * grad_c_alpha
            
            mlc_replace_cond_inplace(extra_args['cond'], c.squeeze(dim=0))
        
        # noise step alike ancestral
        if i <= n_steps - 1:
            x = x + torch.randn_like(x) * 1e-5

    # post-optimizing image latent
    if grad_x_iter:
        c = mlc_get_cond(extra_args['cond']).unsqueeze(dim=0)

        for i in trange(grad_x_iter):
            if state.interrupted: break

            with torch.enable_grad():
                c = c.detach().clone() ; c.requires_grad = True
                x = x.detach().clone() ; x.requires_grad = True

                loss = get_latent_loss(sd_model, x, t, c, noise)
                grad = torch.autograd.grad(loss, inputs=x, grad_outputs=loss)[0]

            with torch.no_grad():
                print(f'loss_x: {loss.mean().item():.7f}, grad_x: {grad.mean().item():.7f}')

            x = x.detach() - grad.sign() * grad_x_alpha

    return x

@torch.no_grad()
def sample_euler_ex(model:CFGDenoiser, x:Tensor, sigmas:List, extra_args={}, callback=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    momentum           = settings['momentum']
    momentum_sign      = settings['momentum_sign']
    momentum_hist      = settings['momentum_hist']
    momentum_hist_init = settings['momentum_hist_init']

    if   momentum_hist_init == 'zero':      history_d = 0
    elif momentum_hist_init == 'rand_init': history_d = x
    elif momentum_hist_init == 'rand_new':  history_d = torch.randn_like(x)
    else: raise ValueError(f'unknown momentum_hist_init: {momentum_hist_init}')

    s_in = x.new_ones([x.shape[0]])
    n_steps = len(sigmas) - 1
    for i in trange(n_steps):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0: x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat

        if 'momentum step':
            # decide correct direction
            sign = momentum_sign
            action_pool = ['pos', 'neg']
            if   sign == 'rand'   : sign = random.choice(action_pool)
            elif sign == 'pos_neg': sign = action_pool[int(i < n_steps // 2)]
            elif sign == 'neg_pos': sign = action_pool[int(i > n_steps // 2)]
            else: pass

            # correct current `d` with momentum
            p = 1.0 - momentum
            if   sign == 'pos': momentum_d = (1.0 - p) * d + p * history_d
            elif sign == 'neg': momentum_d = (1.0 + p) * d - p * history_d
            else: raise ValueError(f'unknown momentum sign {sign}')
            
            # Euler method with momentum
            x = x + momentum_d * dt

            # update momentum history
            q = 1.0 - momentum_hist
            if (isinstance(history_d, int) and history_d == 0):
                history_d = momentum_d
            else:
                if   sign == 'pos': history_d = (1.0 - q) * history_d + q * momentum_d
                elif sign == 'neg': history_d = (1.0 + q) * history_d - q * momentum_d
                else: raise ValueError(f'unknown momentum sign {sign}')
        else:
            # Euler method original
            x = x + d * dt

    return x

@torch.no_grad()
def sample_euler_ancestral_ex(model:CFGDenoiser, x:Tensor, sigmas:List, extra_args={}, callback=None, eta=1.):
    momentum           = settings['momentum']
    momentum_sign      = settings['momentum_sign']
    momentum_hist      = settings['momentum_hist']
    momentum_hist_init = settings['momentum_hist_init']

    # x: [1, 4, 64, 64], 外源的高斯随机噪声
    show_featmap(x, 'before sample')

    # 记录梯度历史的惯性
    if   momentum_hist_init == 'zero':      history_d = 0
    elif momentum_hist_init == 'rand_init': history_d = x
    elif momentum_hist_init == 'rand_new':  history_d = torch.randn_like(x)
    else: raise ValueError(f'unknown momentum_hist_init: {momentum_hist_init}')

    s_in = x.new_ones([x.shape[0]])     # [B=1]
    n_steps = len(sigmas) - 1
    for i in trange(n_steps):
        # [1, 4, 64, 64], 一步降噪后，生成的图越来越明显, sigma理解为步长dx
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        show_featmap(denoised, f'denoised (step {i})')

        # 噪声差分 eps 应当服从正太分布
        eps = denoised - x
        show_featmap(eps, f'eps (step {i})')

        # scalar, sigma_down < sigma_up < sigmas[i + 1] < sigmas[i]
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        # # [1, 4, 64, 64], 梯度 = dy/dx = (x - denoised) / sigmas[i]
        d = to_d(x, sigmas[i], denoised)
        show_featmap(d, f'd (step {i}); sigma={sigmas[i]}')

        # ancestral scheduling (down)
        dt = sigma_down - sigmas[i]        # scalar, 沿着梯度移动的步长
        if 'momentum step':
            # decide correct direction
            sign = momentum_sign
            action_pool = ['pos', 'neg']
            if   sign == 'rand'   : sign = random.choice(action_pool)
            elif sign == 'pos_neg': sign = action_pool[int(i < n_steps // 2)]
            elif sign == 'neg_pos': sign = action_pool[int(i > n_steps // 2)]
            else: pass

            # correct current `d` with momentum
            p = 1.0 - momentum
            if   sign == 'pos': momentum_d = (1.0 - p) * d + p * history_d
            elif sign == 'neg': momentum_d = (1.0 + p) * d - p * history_d
            else: raise ValueError(f'unknown momentum sign {sign}')
            
            # Euler method with momentum
            x = x + momentum_d * dt

            # update momentum history
            q = 1.0 - momentum_hist
            if (isinstance(history_d, int) and history_d == 0):
                history_d = momentum_d
            else:
                if   sign == 'pos': history_d = (1.0 - q) * history_d + q * momentum_d
                elif sign == 'neg': history_d = (1.0 + q) * history_d - q * momentum_d
                else: raise ValueError(f'unknown momentum sign {sign}')
        else:
            # Euler method original
            x = x + d * dt
        show_featmap(x, f'x + d x dt (step {i}); sigma_down={sigma_down:.4f}')    # 作画内容逐渐显露
        
        # ancestral scheduling (up)
        x = x + torch.randn_like(x) * sigma_up
        show_featmap(x, f'x + randn (step {i}); sigma_up={sigma_up:.4f}')         # 再被压抑下去
    
    # x: [1, 4, 64, 64], 采样后是否有语义：有，就是生成图的小图，所以vae decoder做的事基本就是超分
    show_featmap(x, 'after sample')

    # optimze cond to fixed noise
    # cond = mlc_get_cond(extra_args['cond'])
    #loss_cond = get_latent_loss(sd_model, denoised, cond, noise=last_noise)

    # optimze latent delta to zero, `model` should be identity on `x``
    #x_hat = model(x, sigmas[i] * s_in, **extra_args)
    #loss_latent = F.mse_loss(x_hat, x)

    return x

# ↑↑↑ the above is modified from 'k_diffusion/sampling.py' ↑↑↑


# ↓↓↓ the following is modified from 'modules/sd_samplers.py' ↓↓↓

all_samplers_sonar = [
    # wrap the well-known samplers
    SamplerData('Euler a', lambda model: KDiffusionSamplerHijack(model, 'sample_euler_ancestral_ex'), ['k_euler_a_ex'], {}),
    SamplerData('Euler',   lambda model: KDiffusionSamplerHijack(model, 'sample_euler_ex'),           ['k_euler_ex'],   {}),
    # my dev-playground
    SamplerData('Naive',   lambda model: KDiffusionSamplerHijack(model, 'sample_naive_ex'),           ['naive_ex'],     {}),
]
sampler_extra_params_sonar = {
    # 'sampler_name': ['param1', 'param2', ...]
    'sample_euler_ex': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
}
CHOICE_SAMPLER = [s.name for s in all_samplers_sonar]

def create_sampler(sd_model):
    for config in all_samplers_sonar:
        if config.name == settings['sampler']:
            sampler = config.constructor(sd_model)
            sampler.config = config
            return sampler

    raise ValueError(f'implementaion of sampler {settings["sampler"]!r} not found')

class KDiffusionSamplerHijack(KDiffusionSampler):

    def __init__(self, sd_model, funcname):
        # init the homogenus base sampler
        super().__init__('sample_euler', sd_model)      # the 'funcname' this is dummy
        # hijack the sampler object
        self.funcname = funcname
        self.func = globals().get(self.funcname)
        self.extra_params = sampler_extra_params_sonar.get(funcname, [])

    def get_sigmas(self, p:StableDiffusionProcessing, steps:int) -> List[float]:
        if p.sampler_noise_scheduler_override:
            sigmas = p.sampler_noise_scheduler_override(steps)
        elif self.config is not None and self.config.options.get('scheduler', None) == 'karras':
            sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=0.1, sigma_max=10, device=shared.device)
        else:
            # self.model_wrap: k_diffusion.external.CompVisDenoiser
            sigmas = self.model_wrap.get_sigmas(steps)
        return sigmas
    
    def sample(self, p:StableDiffusionProcessing, x:Tensor, 
               conditioning:MulticondLearnedConditioning, unconditional_conditioning:ScheduledPromptConditioning, 
               steps:int=None, image_conditioning:Tensor=None):

        steps = steps or p.steps
        # sigmas: [16=steps+1], sigma[0]=14.6116 && sigma[-1]=0.0 都是常量，中间是递减的插值(指数衰减？)
        sigmas = self.get_sigmas(p, steps)
        # x: [B=1, C=4, H=64, W=64]
        x = x * sigmas[0]

        extra_params_kwargs = self.initialize(p)
        if 'sigma_min' in inspect.signature(self.func).parameters:
            extra_params_kwargs['sigma_min'] = self.model_wrap.sigmas[0].item()
            extra_params_kwargs['sigma_max'] = self.model_wrap.sigmas[-1].item()
            if 'n' in inspect.signature(self.func).parameters:
                extra_params_kwargs['n'] = steps
        else:
            extra_params_kwargs['sigmas'] = sigmas

        self.last_latent = x        # [1, 4, 64, 64]

        samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, x, extra_args={
            'cond': conditioning,                   # prompt cond
            'image_cond': image_conditioning,       # [1, 5, 1, 1], dummy
            'uncond': unconditional_conditioning,   # negaivte prompt cond
            'cond_scale': p.cfg_scale               # 7.0
        }, callback=self.callback_state, **extra_params_kwargs))

        return samples

    def sample_img2img(self, p:StableDiffusionProcessing, x:Tensor, noise:Tensor, 
                       conditioning:MulticondLearnedConditioning, unconditional_conditioning:ScheduledPromptConditioning, 
                       steps:int=None, image_conditioning:Tensor=None):
        
        steps, t_enc = setup_img2img_steps(p, steps)
        sigmas = self.get_sigmas(p, steps)
        sigma_sched = sigmas[steps - t_enc - 1:]
        xi = x + noise * sigma_sched[0]
        
        extra_params_kwargs = self.initialize(p)
        if 'sigma_min' in inspect.signature(self.func).parameters:
            ## last sigma is zero which isn't allowed by DPM Fast & Adaptive so taking value before last
            extra_params_kwargs['sigma_min'] = sigma_sched[-2]
        if 'sigma_max' in inspect.signature(self.func).parameters:
            extra_params_kwargs['sigma_max'] = sigma_sched[0]
        if 'n' in inspect.signature(self.func).parameters:
            extra_params_kwargs['n'] = len(sigma_sched) - 1
        if 'sigma_sched' in inspect.signature(self.func).parameters:
            extra_params_kwargs['sigma_sched'] = sigma_sched
        if 'sigmas' in inspect.signature(self.func).parameters:
            extra_params_kwargs['sigmas'] = sigma_sched

        self.model_wrap_cfg.init_latent = x
        self.last_latent = x

        samples = self.launch_sampling(t_enc + 1, lambda: self.func(self.model_wrap_cfg, xi, extra_args={
            'cond': conditioning, 
            'image_cond': image_conditioning, 
            'uncond': unconditional_conditioning, 
            'cond_scale': p.cfg_scale
        }, callback=self.callback_state, **extra_params_kwargs))

        return samples

# ↑↑↑ the above is modified from 'modules/sd_samplers.py' ↑↑↑


def get_latent_loss(self:LatentDiffusion, x:Tensor, t:Tensor, c:Tensor, noise:Tensor) -> Tensor:
    # 这个t是时间嵌入(time embed)，决定了降噪程度(似乎是逆向sigma调度)，貌似对结果影响不大
    t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long() if t is None else t  # [B=1] 

    x_start = x
    noise = torch.randn_like(x_start) if noise is None else noise   # [B=1, C=4, H=64, W=64]
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)      # [B=1, C=4, H=64, W=64], diffsusion step
    model_output = self.apply_model(x_noisy, t, c)                  # [B=1, C=4, H=64, W=64], inverse diffusion step

    if    self.parameterization == "x0":  target = x_start
    elif  self.parameterization == "eps": target = noise            # it goes this way
    else: raise NotImplementedError()

    loss_simple = self.get_loss(model_output, target, mean=False)   # self.loss_type == 'l2'; the model shoud predict the noise that is added
    logvar_t = self.logvar[t].to(self.device)                       # self.logvar == torch.zeros([1000])
    loss = loss_simple / torch.exp(logvar_t) + logvar_t
    loss = self.l_simple_weight * loss                              # self.l_simple_weight == 1.0

    loss_vlb = self.get_loss(model_output, target, mean=False)
    loss_vlb = (self.lvlb_weights[t] * loss_vlb)                    # self.lvlb_weights is non-zeros
    loss += (self.original_elbo_weight * loss_vlb)                  # but self.original_elbo_weight == 0.0, I don't know why :(

    return loss                                                     # [B=1, C=4, H=64, W=64]

def image_to_latent(model:LatentDiffusion, img: Image) -> Tensor:
    im = np.array(img).astype(np.uint8)
    im = (im / 127.5 - 1.0).astype(np.float32)
    x = torch.from_numpy(im)
    x = torch.moveaxis(x, 2, 0)
    x = x.unsqueeze(dim=0)          # [B=1, C=3, H=512, W=512]
    x = x.to(model.device)
    
    latent = model.get_first_stage_encoding(model.encode_first_stage(x))    # [B=1, C=4, H=64, W=64]
    return latent

def mlc_get_cond(c:MulticondLearnedConditioning) -> Tensor:
    return c.batch[0][0].schedules[0].cond      # [B=1, T=77, D=768]

def mlc_replace_cond_inplace(c:MulticondLearnedConditioning, cond: Tensor):
    spc = c.batch[0][0].schedules[0]
    c.batch[0][0].schedules[0] = ScheduledPromptConditioning(spc.end_at_step, cond)


class Script(scripts.Script):

    def title(self):
        return 'Sonar'

    def describe(self):
        return "Wrapped samplers with tricks to optimize prompt condition and image latent for better image quality"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        sampler = gr.Radio(label='Base Sampler', value=lambda: DEFAULT_SAMPLER, choices=CHOICE_SAMPLER)

        with gr.Group() as tab_momentum:
            with gr.Row():
                momentum      = gr.Slider(label='Momentum (current)', minimum=0.75, maximum=1.0, value=lambda: DEFAULT_MOMENTUM)
                momentum_hist = gr.Slider(label='Momentum (history)', minimum=0.0,  maximum=1.0, value=lambda: DEFAULT_MOMENTUM_HIST)
            with gr.Row():
                momentum_sign      = gr.Radio(label='Momentum sign',         value=lambda: DEFAULT_MOMENTUM_SIGN,      choices=CHOICE_MOMENTUM_SIGN)
                momentum_hist_init = gr.Radio(label='Momentum history init', value=lambda: DEFAULT_MOMENTUM_HIST_INIT, choices=CHOICE_MOMENTUM_HIST_INIT)
        
        with gr.Group(visible=False) as tab_gradient:
            with gr.Row():
                grad_c_iter  = gr.Slider(label='Optimize cond step count', value=lambda: DEFAULT_GRAD_C_ITER,  minimum=0,     maximum=10,   step=1)
                grad_c_alpha = gr.Slider(label='Optimize cond step size',  value=lambda: DEFAULT_GRAD_C_ALPHA, minimum=-0.01, maximum=0.01, step=0.001)
                grad_c_skip  = gr.Slider(label='Skip the first n-steps',   value=lambda: DEFAULT_GRAD_C_SKIP,  minimum=0,     maximum=100,  step=1)
            with gr.Row():
                grad_x_iter  = gr.Slider(label='Optimize latent step count', value=lambda: DEFAULT_GRAD_X_ITER,  minimum=0,    maximum=40,  step=1)
                grad_x_alpha = gr.Slider(label='Optimize latent step size',  value=lambda: DEFAULT_GRAD_X_ALPHA, minimum=-0.1, maximum=0.1, step=0.01)
                grad_fuzzy   = gr.Checkbox(label='Fuzzy grad',               value=lambda: DEFAULT_GRAD_FUZZY)

        with gr.Group(visible=False) as tab_file:
            with gr.Row():
                ref_meth = gr.Radio(label='Guide step method', value=lambda: DEFAULT_REF_METH, choices=CHOICE_REF_METH)
                ref_hgf = gr.Slider(label='Guide factor', value=lambda: DEFAULT_REF_HGF, minimum=-1, maximum=1, step=0.001)
                ref_min_step = gr.Number(label='Ref start step', value=lambda: DEFAULT_REF_MIN_STEP)
                ref_max_step = gr.Number(label='Ref stop step', value=lambda: DEFAULT_REF_MAX_STEP)
            with gr.Row():
                ref_img = gr.File(label='Reference image file', interactive=True)

        def swith_sampler(sampler:str):
            SHOW_TABS = {
                # (show_momt, show_grad, show_file)
                'Euler a': (True, False, False),
                'Euler':   (True, False, False),
                'Naive':   (True, True,  True),
            }
            show_momt, show_grad, show_file = SHOW_TABS[sampler]
            return [   
                { 'visible': show_momt, '__type__': 'update' },
                { 'visible': show_grad, '__type__': 'update' },
                { 'visible': show_file, '__type__': 'update' },
            ]

        sampler.change(swith_sampler, inputs=[sampler], outputs=[tab_momentum, tab_gradient, tab_file])

        return [sampler, 
                momentum, momentum_hist, momentum_hist_init, momentum_sign, 
                grad_c_iter, grad_c_alpha, grad_c_skip, grad_x_iter, grad_x_alpha, grad_fuzzy,
                ref_meth, ref_hgf, ref_min_step, ref_max_step, ref_img]
    
    def run(self, p:StableDiffusionProcessing, sampler:str, 
            momentum:float, momentum_hist:float, momentum_hist_init:str, momentum_sign:str,
            grad_c_iter:int, grad_c_alpha:float, grad_c_skip:int, grad_x_iter:int, grad_x_alpha:float, grad_fuzzy:bool,
            ref_meth:str, ref_hgf:float, ref_min_step:float, ref_max_step:float, ref_img:object):
        
        # save settings to global
        settings['sampler']            = sampler
        settings['momentum']           = momentum
        settings['momentum_hist']      = momentum_hist
        settings['momentum_hist_init'] = momentum_hist_init
        settings['momentum_sign']      = momentum_sign
        settings['grad_c_iter']        = grad_c_iter
        settings['grad_c_alpha']       = grad_c_alpha
        settings['grad_c_skip']        = int(grad_c_skip)
        settings['grad_x_iter']        = grad_x_iter
        settings['grad_x_alpha']       = grad_x_alpha
        settings['grad_fuzzy']         = grad_fuzzy
        settings['ref_meth']           = ref_meth
        settings['ref_min_step']       = int(ref_min_step) if ref_min_step > 1 else round(ref_min_step * p.steps)
        settings['ref_max_step']       = int(ref_max_step) if ref_max_step > 1 else round(ref_max_step * p.steps)
        settings['ref_hgf']            = ref_hgf
        settings['ref_img']            = ref_img

        #pp(settings)

        state.job_count = p.n_iter * p.batch_size
        proc = process_images(p)

        return Processed(p, proc.images, p.seed, proc.info)
