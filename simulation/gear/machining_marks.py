"""Object-space machining marks for Mitsuba's bump-map adapter.

Import after selecting a Mitsuba variant. Geometry coordinates are millimeters;
the height field is in scene meters. Spacing and amplitude are estimates.
"""

import math

import drjit as dr
import mitsuba as mi
import numpy as np


def smooth_ramp(value, low, high):
    t = dr.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3 - 2 * t), 6 * t * (1 - t) / (high - low)


class MachiningMarks(mi.Texture):
    def __init__(self, props):
        super().__init__(props)
        # For a moved/rotated gear, supply its complete inverse object transform.
        transform = props.get("to_model", mi.ScalarTransform4f().scale(1000))
        # Explicit Properties transforms can arrive as double-precision scalars.
        self.to_model = mi.Transform4f(np.asarray(transform.matrix, dtype=np.float32))
        self.spacing = props.get("spacing_mm")
        self.amplitude = props.get("amplitude_mm") * 0.001
        self.jitter = props.get("jitter_mm")
        self.jitter_length = props.get("jitter_length_mm")
        self.phase_variation = props.get("phase_variation")
        self.flank_strength = props.get("flank_strength")
        self.inner_radius = props.get("inner_radius_mm")
        self.outer_radius = props.get("outer_radius_mm")
        self.front_z = props.get("front_z_mm")
        self.outer_root_z = props.get("outer_root_z_mm")
        self.tooth_height = props.get("tooth_height_mm")
        self.teeth = props.get("tooth_count")
        self.mask_only = props.get("mask_only", False)

    def surface_mask(self, si):
        p = self.to_model @ si.p
        r = dr.sqrt(p.x * p.x + p.y * p.y)
        slope = (self.outer_root_z - self.front_z) / (self.outer_radius - self.inner_radius)
        above_root = p.z - (self.front_z + slope * (r - self.inner_radius))
        radial_mask, _ = smooth_ramp(r, self.inner_radius - 0.05, self.inner_radius)
        height_mask, _ = smooth_ramp(above_root, 0.01, 0.15)
        return radial_mask * height_mask

    def height_and_gradient(self, si):
        p = self.to_model @ si.p
        r2 = dr.maximum(p.x * p.x + p.y * p.y, 1e-8)
        r = dr.sqrt(r2)
        radial_gradient = mi.Vector3f(p.x / r, p.y / r, 0)
        angle_gradient = mi.Vector3f(-p.y / r2, p.x / r2, 0)
        theta = dr.atan2(p.y, p.x)
        angular = dr.sin(self.teeth * theta)
        angular_gradient = self.teeth * dr.cos(self.teeth * theta) * angle_gradient

        # Fine transverse strokes, with small smooth deviations from a perfect
        # periodic pattern. These variations are illustrative, not measurements.
        jitter_phase = 2 * math.pi * r / self.jitter_length + 0.4 * angular
        jitter_gradient = 2 * math.pi / self.jitter_length * radial_gradient + 0.4 * angular_gradient
        coordinate = r + self.jitter * dr.sin(jitter_phase)
        coordinate_gradient = radial_gradient + self.jitter * dr.cos(jitter_phase) * jitter_gradient
        phase = 2 * math.pi * coordinate / self.spacing + self.phase_variation * angular
        phase_gradient = 2 * math.pi / self.spacing * coordinate_gradient + self.phase_variation * angular_gradient
        height = self.amplitude * (dr.sin(phase) + 0.22 * dr.sin(2 * phase + 0.9)) / 1.22
        gradient = self.amplitude * (dr.cos(phase) + 0.44 * dr.cos(2 * phase + 0.9)) / 1.22 * phase_gradient

        # Restrict this particular reference pattern to the teeth. Its strength
        # increases near the crest and fades toward the root and inner edge.
        slope = (self.outer_root_z - self.front_z) / (self.outer_radius - self.inner_radius)
        above_root = p.z - (self.front_z + slope * (r - self.inner_radius))
        above_gradient = mi.Vector3f(0, 0, 1) - slope * radial_gradient
        rise, rise_derivative = smooth_ramp(above_root, 0.15, 0.7)
        crest, crest_derivative = smooth_ramp(above_root, self.tooth_height - 0.7, self.tooth_height - 0.1)
        radial_mask, radial_derivative = smooth_ramp(r, self.inner_radius, self.inner_radius + 0.15)
        weight = self.flank_strength + (1 - self.flank_strength) * crest
        mask = radial_mask * rise * weight
        mask_gradient = (
            radial_derivative * radial_gradient * rise * weight
            + radial_mask * (rise_derivative * weight + rise * (1 - self.flank_strength) * crest_derivative)
            * above_gradient
        )
        return height * mask, gradient * mask + height * mask_gradient

    def eval_1(self, si, active=True):
        if self.mask_only:
            return dr.select(active, self.surface_mask(si), 0)
        height, _ = self.height_and_gradient(si)
        return dr.select(active, height, 0)

    def eval_1_grad(self, si, active=True):
        _, gradient = self.height_and_gradient(si)
        # Chain rule converts the object-space gradient into derivatives along
        # Mitsuba's surface tangents. This works without adding UVs to the OBJ.
        tangent_u = self.to_model @ si.dp_du
        tangent_v = self.to_model @ si.dp_dv
        result = mi.Vector2f(dr.dot(gradient, tangent_u), dr.dot(gradient, tangent_v))
        return dr.select(active, result, mi.Vector2f(0))

    def eval(self, si, active=True):
        return mi.UnpolarizedSpectrum(self.eval_1(si, active))

    def eval_3(self, si, active=True):
        return mi.Color3f(self.eval_1(si, active))

    def mean(self):
        return 0.5 if self.mask_only else 0.0

    def max(self):
        return 1.0 if self.mask_only else self.amplitude

    def is_spatially_varying(self):
        return True

    def to_string(self):
        return f"MachiningMarks[spacing={self.spacing} mm, assumed amplitude={self.amplitude * 1000} mm]"


mi.register_texture("gear_machining_marks", lambda props: MachiningMarks(props))


def machining_material(params):
    supplied, estimated, detail = params["supplied"], params["estimated"], params["surface_detail"]
    texture = {
        "type": "gear_machining_marks",
        "spacing_mm": detail["estimated_mark_spacing_mm"],
        "amplitude_mm": detail["assumed_bump_amplitude_mm"],
        "jitter_mm": detail["assumed_spacing_jitter_mm"],
        "jitter_length_mm": detail["assumed_jitter_length_mm"],
        "phase_variation": detail["assumed_cross_tooth_phase_variation"],
        "flank_strength": detail["assumed_flank_strength_fraction"],
        "inner_radius_mm": estimated["tooth_inner_radius"],
        "outer_radius_mm": supplied["outside_diameter"] / 2,
        "front_z_mm": supplied["total_thickness"] - estimated["tooth_height"],
        "outer_root_z_mm": estimated["outer_root_height"],
        "tooth_height_mm": estimated["tooth_height"],
        "tooth_count": supplied["tooth_count"],
    }
    base = {
        "type": "roughconductor", "material": "none",
        "distribution": "ggx", "alpha": 0.27,
        "specular_reflectance": {"type": "rgb", "value": [0.36, 0.39, 0.42]},
    }
    return {
        "type": "blendbsdf",
        "weight": {**texture, "mask_only": True},
        "bsdf_0": base,
        "bsdf_1": {
            "type": "bumpmap", "scale": 1.0, "texture": texture, "bsdf": base,
        },
    }
