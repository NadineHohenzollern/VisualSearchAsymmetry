import math
from typing import Optional

import tensorflow as tf


class EccentricityAvgPool2D(tf.keras.layers.Layer):

    def __init__(
        self,
        ecc_slope: float = 0.2,
        fovea_size: float = 2.0,
        deg2px: float = 30.0,
        foveal_rf: int = 1,
        stride: int = 1,
        **kwargs,
    ):
        """
        Eccentricity-dependent average pooling layer

        Parameters
        ----------
        ecc_slope : float
            Slope of the linear function that defines the pooling diameter as a
            function of eccentricity.
        fovea_size : float
            Size of the fovea in degrees of visual angle.
        deg2px : float
            Conversion factor from degrees of visual angle to pixels.
        foveal_rf : int
            Size of the receptive field at the fovea in pixels.
        stride : int
            Stride of the pooling operation.

        Notes
        -----
        Input shape is (B, H, W, C). H and W must be statically known.
        Fixation has shape (B, 2) as (row, col) in pixels and may be float.
        Fixation and eccentricity use pixel-center coordinates, pooling windows use edge coordinates.
        """
        super().__init__(**kwargs)
        self.ecc_slope = ecc_slope
        self.fovea_size = fovea_size
        self.deg2px = deg2px
        self.foveal_rf = foveal_rf
        self.stride = stride

    def call(
        self,
        x: tf.Tensor,
        fixation: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        if H is None or W is None:
            raise ValueError("EccentricityAvgPool2D requires static H and W.")
        B = tf.shape(x)[0]
        Ho, Wo = math.ceil(H / self.stride), math.ceil(W / self.stride)

        # Summed-area table with a zero row and column, flattened to (B, (H+1)(W+1), C)
        S = tf.cumsum(tf.cumsum(x, axis=1), axis=2)
        S = tf.pad(S, [[0, 0], [1, 0], [1, 0], [0, 0]])
        S = tf.reshape(S, [B, (H + 1) * (W + 1), C])

        y0, y1, x0, x1 = self._get_pooling_windows(x, fixation)

        def corner(yy, xx):
            idx = tf.reshape(yy * (W + 1) + xx, [B, Ho * Wo])
            return tf.gather(S, idx, batch_dims=1)  # (B, Ho*Wo, C)

        total = corner(y1, x1) - corner(y0, x1) - corner(y1, x0) + corner(y0, x0)
        area = tf.cast(tf.reshape((y1 - y0) * (x1 - x0), [B, Ho * Wo, 1]), x.dtype)

        return tf.reshape(total / area, [B, Ho, Wo, C])

    def _get_pooling_windows(
        self,
        x: tf.Tensor,
        fixation: Optional[tf.Tensor] = None,
    ):
        """
        Returns y0, y1, x0, x1 as int32 tensors of shape (B, Ho, Wo) holding the
        edge coordinates of each pooling window.
        """
        H, W = x.shape[1], x.shape[2]
        B = tf.shape(x)[0]
        s = self.stride
        Ho, Wo = math.ceil(H / s), math.ceil(W / s)

        # Block extents in edge coordinates, clamped for the last partial block
        by0 = tf.range(Ho) * s
        bx0 = tf.range(Wo) * s
        by1 = tf.minimum(by0 + s, H)
        bx1 = tf.minimum(bx0 + s, W)

        # Block centers in pixel-center coordinates (used for eccentricity)
        ys = tf.reshape(tf.cast(by0 + by1 - 1, tf.float32) / 2.0, [1, Ho, 1])
        xs = tf.reshape(tf.cast(bx0 + bx1 - 1, tf.float32) / 2.0, [1, 1, Wo])

        if fixation is None:
            center = tf.constant([[(H - 1) / 2.0, (W - 1) / 2.0]], dtype=tf.float32)
            fixation = tf.tile(center, [B, 1])

        fixation = tf.cast(fixation, tf.float32)
        fy = tf.reshape(fixation[:, 0], [B, 1, 1])
        fx = tf.reshape(fixation[:, 1], [B, 1, 1])

        # Euclidean distance from fixation to each pooling window center
        ecc = tf.sqrt((ys - fy) ** 2 + (xs - fx) ** 2)

        # Window diameter, then half-width, never narrower than one stride block
        d = self.foveal_rf + self.ecc_slope * tf.maximum(
            ecc - 0.5 * self.fovea_size * self.deg2px, 0.0
        )
        r = tf.maximum(0.5 * d, s / 2.0)

        # Edge coordinates are pixel center + 0.5, window is [c - r, c + r]
        yc, xc = ys + 0.5, xs + 0.5
        y0 = tf.cast(tf.maximum(tf.floor(yc - r + 0.5), 0.0), tf.int32)
        y1 = tf.cast(tf.minimum(tf.floor(yc + r + 0.5), float(H)), tf.int32)
        x0 = tf.cast(tf.maximum(tf.floor(xc - r + 0.5), 0.0), tf.int32)
        x1 = tf.cast(tf.minimum(tf.floor(xc + r + 0.5), float(W)), tf.int32)

        return y0, y1, x0, x1

    def compute_output_shape(self, input_shape):
        B, H, W, C = input_shape
        return (B, math.ceil(H / self.stride), math.ceil(W / self.stride), C)

    def get_config(self):
        config = super().get_config()
        config.update(
            ecc_slope=self.ecc_slope,
            fovea_size=self.fovea_size,
            deg2px=self.deg2px,
            foveal_rf=self.foveal_rf,
            stride=self.stride,
        )
        return config
    