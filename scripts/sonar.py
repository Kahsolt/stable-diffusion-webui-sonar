import os
from copy import deepcopy
from PIL import Image
from typing import List, Tuple, Union

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np

import modules.scripts as scripts
from modules.shared import state

DEFAULT_SAMPLER            = 'Euler a mg'
DEFAULT_MOMENTUM           = 0.95
DEFAULT_MOMENTUM_HIST      = 0.75
DEFAULT_MOMENTUM_HIST_INIT = 'zero'
DEFAULT_MOMENTUM_SIGN      = 'pos'
DEFAULT_DEBUG              = True

CHOICE_MOMENTUM_SIGN      = ['pos', 'neg', 'pos_neg', 'neg_pos', 'rand']
CHOICE_MOMENTUM_HIST_INIT = ['zero', 'rand_init', 'rand_new']

# the current setting (the wrappers are too deep, we pass it by global var)
settings = {
    'sampler':            DEFAULT_SAMPLER,
    'momentum':           DEFAULT_MOMENTUM,
    'momentum_hist':      DEFAULT_MOMENTUM_HIST,
    'momentum_hist_init': DEFAULT_MOMENTUM_HIST_INIT,
    'momentum_sign':      DEFAULT_MOMENTUM_SIGN,
}

Tensor = torch.Tensor

# ↓↓↓ the following is modified from 'modules/processing.py' ↓↓↓

from modules.processing import *

def process_images(p: StableDiffusionProcessing) -> Processed:
    stored_opts = {k: opts.data[k] for k in p.override_settings.keys()}

    try:
        for k, v in p.override_settings.items():
            setattr(opts, k, v) # we don't call onchange for simplicity which makes changing model, hypernet impossible

        res = process_images_inner(p)

    finally:
        for k, v in stored_opts.items():
            setattr(opts, k, v)

    return res

def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    with open(os.path.join(shared.script_path, "params.txt"), "w", encoding="utf8") as file:
        processed = Processed(p, [], p.seed, "")
        file.write(processed.infotext(p, 0))

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    model_hijack.apply_circular(p.tiling)
    model_hijack.clear_comments()

    comments = {}

    shared.prompt_styles.apply_styles(p)

    if type(p.prompt) == list:
        p.all_prompts = p.prompt
    else:
        p.all_prompts = p.batch_size * p.n_iter * [p.prompt]

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
            seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if len(prompts) == 0:
                break

            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

            with devices.autocast():
                uc = prompt_parser.get_learned_conditioning(shared.sd_model, len(prompts) * [p.negative_prompt], p.steps)
                c = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, prompts, p.steps)

            if len(model_hijack.comments) > 0:
                for comment in model_hijack.comments:
                    comments[comment] = 1

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            with devices.autocast():
                #samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)
                if   isinstance(p, StableDiffusionProcessingTxt2Img): sample_func = StableDiffusionProcessingTxt2Img_sample
                elif isinstance(p, StableDiffusionProcessingImg2Img): sample_func = StableDiffusionProcessingImg2Img_sample
                else: raise ValueError
                samples_ddim = sample_func(p, conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)
            
            samples_ddim = samples_ddim.to(devices.dtype_vae)
            x_samples_ddim = decode_first_stage(p.sd_model, samples_ddim)
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

    res = Processed(p, output_images, p.all_seeds[0], infotext() + "".join(["\n\n" + x for x in comments]), subseed=p.all_subseeds[0], all_prompts=p.all_prompts, all_seeds=p.all_seeds, all_subseeds=p.all_subseeds, index_of_first_image=index_of_first_image, infotexts=infotexts)

    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res

def StableDiffusionProcessingTxt2Img_sample(self:StableDiffusionProcessingTxt2Img, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
    # hijack the sampler~
    self.sampler = create_sampler(self.sd_model)

    if not self.enable_hr:
        x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))
        return samples

    x = create_random_tensors([opt_C, self.firstphase_height // opt_f, self.firstphase_width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
    samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x, self.firstphase_width, self.firstphase_height))

    samples = samples[:, :, self.truncate_y//2:samples.shape[2]-self.truncate_y//2, self.truncate_x//2:samples.shape[3]-self.truncate_x//2]

    """saves image before applying hires fix, if enabled in options; takes as an arguyment either an image or batch with latent space images"""
    def save_intermediate(image, index):
        if not opts.save or self.do_not_save_samples or not opts.save_images_before_highres_fix:
            return

        if not isinstance(image, Image.Image):
            image = sd_samplers.sample_to_image(image, index)

        images.save_image(image, self.outpath_samples, "", seeds[index], prompts[index], opts.samples_format, suffix="-before-highres-fix")

    if opts.use_scale_latent_for_hires_fix:
        for i in range(samples.shape[0]):
            save_intermediate(samples, i)

        samples = torch.nn.functional.interpolate(samples, size=(self.height // opt_f, self.width // opt_f), mode="bilinear")

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

            image = images.resize_image(0, image, self.width, self.height)
            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            batch_images.append(image)

        decoded_samples = torch.from_numpy(np.array(batch_images))
        decoded_samples = decoded_samples.to(shared.device)
        decoded_samples = 2. * decoded_samples - 1.

        samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))

        image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)

    shared.state.nextjob()

    # hijack the sampler~
    self.sampler = create_sampler(self.sd_model)

    noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)

    # GC now before running the next img2img to prevent running out of memory
    x = None
    devices.torch_gc()

    samples = self.sampler.sample_img2img(self, samples, noise, conditioning, unconditional_conditioning, steps=self.steps, image_conditioning=image_conditioning)

    return samples

def StableDiffusionProcessingImg2Img_sample(self:StableDiffusionProcessingImg2Img, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
    x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)

    # hijack the sampler~
    self.sampler = create_sampler(self.sd_model)
    samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)

    if self.mask is not None:
        samples = samples * self.nmask + self.init_latent * self.mask

    del x
    devices.torch_gc()

    return samples

# ↑↑↑ the above is modified from 'modules/processing.py' ↑↑↑


# ↓↓↓ the following is modified from 'modules/sd_samplers.py' ↓↓↓

from modules.sd_samplers import *
from modules.prompt_parser import ScheduledPromptConditioning, MulticondLearnedConditioning
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

all_samplers_mg = [
    SamplerData('Euler a mg', lambda model: KDiffusionSamplerHijack('sample_euler_ancestral', model, sample_euler_ancestral_momentum_grad), ['k_euler_a_mg'], {}),
    SamplerData('Euler mg',   lambda model: KDiffusionSamplerHijack('sample_euler',           model, sample_euler_momentum_grad),           ['k_euler_mg'],   {}),
    SamplerData('DDIM mg',    lambda model: VanillaStableDiffusionSampler(DDIMSamplerHijack, model), ['ddim_mg'], {}),
    SamplerData('PLMS mg',    lambda model: VanillaStableDiffusionSampler(PLMSSamplerHijack, model), ['plms_mg'], {}),
]
CHOICE_SAMPLER = [s.name for s in all_samplers_mg]

def create_sampler(sd_model):
    for config in all_samplers_mg:
        if config.name == settings['sampler']:
            sampler = config.constructor(sd_model)
            sampler.config = config
            return sampler

    raise ValueError(f'implementaion of sampler {settings["sampler"]!r} not found')

class KDiffusionSamplerHijack(KDiffusionSampler):

    def __init__(self, base_funcname, sd_model, real_func):
        # init the homogenus base sampler
        super().__init__(base_funcname, sd_model)
        # hijack the sampler object
        self.func = real_func

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

        samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, p.sd_model, x, extra_args={
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

        samples = self.launch_sampling(t_enc + 1, lambda: self.func(self.model_wrap_cfg, p.sd_model, xi, extra_args={
            'cond': conditioning, 
            'image_cond': image_conditioning, 
            'uncond': unconditional_conditioning, 
            'cond_scale': p.cfg_scale
        }, callback=self.callback_state, **extra_params_kwargs))

        return samples

# ↑↑↑ the above is modified from 'modules/sd_samplers.py' ↑↑↑


# ↓↓↓ the following is modified from 'k_diffusion/sampling.py' ↓↓↓

from tqdm.auto import trange
from modules.sd_samplers import CFGDenoiser
from ldm.models.diffusion.ddpm import LatentDiffusion

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    # 就是零阶差分，数除sigma以放缩数值
    #return (x - denoised) / append_dims(sigma, x.ndim)
    return (x - denoised) / sigma

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def show_featmap(x, title=''):
    if 'not show': return
    import os
    import matplotlib.pyplot as plt
    IMG_SAVE_PATH = r'C:\Workspace\stable-diffusion-webui-note\img_sampling_process'
    x_np = x.squeeze().cpu().numpy()
    x_np_abs = np.abs(x_np)
    print(f'[{title}]')
    print('   x_np:',     x_np    .max(), x_np    .min(), x_np    .mean(), x_np    .std())
    print('   x_np_abs:', x_np_abs.max(), x_np_abs.min(), x_np_abs.mean(), x_np_abs.std())
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(x_np[i])
    plt.suptitle(title)
    plt.savefig(os.path.join(IMG_SAVE_PATH, f'{title}.png'))
    plt.show()

def sample_euler_ancestral_momentum_grad(model:CFGDenoiser, sd_model:LatentDiffusion, x:Tensor, sigmas:List, extra_args={}, callback=None, eta=1.):
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

    s_in = x.new_ones([x.shape[0]])     # [B=1], 不知道啥用，类型转换？
    n_steps = len(sigmas) - 1
    for i in trange(n_steps):
        # [1, 4, 64, 64], 一步降噪后，生成的图越来越明显
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        show_featmap(denoised, f'denoised (step {i})')

        # 噪声差分 eps
        delta = denoised - x
        show_featmap(delta, f'delta (step {i})')

        # scalar, sigma_down < sigma_up < sigmas[i + 1] < sigmas[i]
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)   # [1, 4, 64, 64], 这应该是某种梯度
        show_featmap(d, f'd (step {i}); sigma={sigmas[i]}')

        # ancestral scheduling (down)
        dt = sigma_down - sigmas[i]        # scalar, 沿着梯度移动的步长
        if 'momentum step':
            # decide correct direction
            sign = momentum_sign
            action_pool = ['pos', 'neg']
            if   sign == 'random' : sign = random.choice(action_pool)
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

def sample_euler_momentum_grad(model:CFGDenoiser, sd_model:LatentDiffusion, x, sigmas, extra_args={}, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x

class DDIMSamplerHijack(DDIMSampler):

    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__(model, schedule, **kwargs)
    
    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

class PLMSSamplerHijack(PLMSSampler):

    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__(model, schedule, **kwargs)

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t

# ↑↑↑ the above is modified from 'k_diffusion/sampling.py' ↑↑↑


def image_to_latent(model, img: Image) -> Tensor:
    #from ldm.models.diffusion import LatentDiffusion
    # type(model) == LatentDiffusion

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

def mlc_replace_cond(c:MulticondLearnedConditioning, cond: Tensor) -> MulticondLearnedConditioning:
    r = deepcopy(c)
    spc = r.batch[0][0].schedules[0]
    r.batch[0][0].schedules[0] = ScheduledPromptConditioning(spc.end_at_step, cond)
    return r


class Script(scripts.Script):

    def title(self):
        return 'Sonar'

    def describe(self):
        return "Tricks to optimize prompt condition and image latent for better image quality"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        sampler = gr.Radio(label='Sampler', value=lambda: DEFAULT_SAMPLER, choices=CHOICE_SAMPLER)

        with gr.Row():
            momentum      = gr.Slider(label='Momentum (current)', minimum=0.75, maximum=1.0, value=lambda: DEFAULT_MOMENTUM)
            momentum_hist = gr.Slider(label='Momentum (history)', minimum=0.0, maximum=1.0, value=lambda: DEFAULT_MOMENTUM_HIST)
        with gr.Row():
            momentum_sign = gr.Radio(label='Momentum Sign', value=lambda: DEFAULT_MOMENTUM_SIGN, choices=CHOICE_MOMENTUM_SIGN)
            momentum_hist_init = gr.Radio(label='Momentum init history', value=lambda: DEFAULT_MOMENTUM_HIST_INIT, choices=CHOICE_MOMENTUM_HIST_INIT)

        return [sampler, momentum,  momentum_hist, momentum_hist_init, momentum_sign]
    
    def run(self, p:StableDiffusionProcessing, sampler:str, 
            momentum:float, momentum_hist:float, momentum_hist_init:str, momentum_sign:str):
        
        # save settings to globa;
        settings['sampler']            = sampler
        settings['momentum']           = momentum
        settings['momentum_hist']      = momentum_hist
        settings['momentum_hist_init'] = momentum_hist_init
        settings['momentum_sign']      = momentum_sign

        state.job_count = p.n_iter * p.batch_size
        proc = process_images(p)

        return Processed(p, proc.images, p.seed, proc.info)
