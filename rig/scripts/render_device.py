"""Select Cycles hardware for Lisa renders, independently of saved preferences."""
import json
import os

import bpy


def configure_cycles_device(scene):
    """Prefer the RTX PRO 2000 with OptiX; log any CPU fallback explicitly."""
    requested = os.environ.get('LISA_RENDER_DEVICE', 'AUTO').upper()
    preferred = os.environ.get('LISA_RENDER_GPU', 'RTX PRO 2000').strip()
    if requested not in ('AUTO', 'OPTIX', 'CUDA', 'CPU'):
        raise ValueError('LISA_RENDER_DEVICE must be AUTO, OPTIX, CUDA or CPU')
    attempts = []
    if requested != 'CPU':
        preferences = bpy.context.preferences.addons['cycles'].preferences
        for backend in (('OPTIX', 'CUDA') if requested == 'AUTO' else (requested,)):
            try:
                preferences.compute_device_type = backend
                devices = preferences.get_devices_for_type(backend)
                available = [device for device in devices if device.type == backend]
                matches = [device for device in available if preferred.lower() in device.name.lower()]
                if preferred and 'LISA_RENDER_GPU' in os.environ and not matches:
                    attempts.append(f'{backend}: requested GPU {preferred!r} not visible')
                    continue
                chosen = (matches or available)[:1]
                if not chosen:
                    attempts.append(f'{backend}: no GPU visible')
                    continue
                selected = chosen[0]
                for device in preferences.devices:
                    device.use = device.type == backend and device.id == selected.id
                scene.cycles.device = 'GPU'
                scene.cycles.denoising_use_gpu = preferences.has_oidn_gpu_devices()
                result = {'device': 'GPU', 'backend': backend, 'gpu': selected.name,
                          'gpu_denoising': scene.cycles.denoising_use_gpu}
                print('LISA_RENDER_DEVICE ' + json.dumps(result), flush=True)
                return result
            except (TypeError, RuntimeError) as error:
                attempts.append(f'{backend}: {error}')
        if requested != 'AUTO':
            raise RuntimeError('GPU rendering unavailable: ' + '; '.join(attempts)
                               + '. Run with NVIDIA device access, outside the restricted sandbox.')
    scene.cycles.device = 'CPU'
    scene.cycles.denoising_use_gpu = False
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4
    result = {'device': 'CPU', 'backend': 'CPU', 'gpu': None,
              'reason': 'explicit CPU request' if requested == 'CPU' else '; '.join(attempts)}
    print('LISA_RENDER_DEVICE ' + json.dumps(result), flush=True)
    if requested == 'AUTO':
        print('GPU unavailable: using CPU. NVIDIA devices may be hidden by the sandbox; '
              'use hardware access or LISA_RENDER_DEVICE=OPTIX to require the GPU.', flush=True)
    return result
