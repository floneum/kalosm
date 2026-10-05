//! The value network, in ordinary Fusor tensor operations: an NNUE-shaped MLP.
use crate::INPUT;
use fusor::{Device, Error, Result, Tensor, tensor::Dyn};

pub const WIDTHS: [usize; 4] = [64, 128, 256, 512];
pub const DEPTHS: [usize; 3] = [2, 3, 4];
pub const HIDDENS: [usize; 3] = [16, 32, 64];

/// The network's size: the first layer's `width` (the accumulator the CPU
/// search updates incrementally), `depth` layers in all, and the `hidden`
/// width of every layer after the first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub width: usize,
    pub depth: usize,
    pub hidden: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            width: 128,
            depth: 2,
            hidden: 32,
        }
    }
}
impl Config {
    pub fn validate(self) -> Result<()> {
        if WIDTHS.contains(&self.width) && DEPTHS.contains(&self.depth) && HIDDENS.contains(&self.hidden) {
            Ok(())
        } else {
            Err(Error::Shape("invalid chess model dimensions".into()))
        }
    }
    /// Every valid size.
    pub fn all() -> impl Iterator<Item = Config> {
        WIDTHS.into_iter().flat_map(|width| {
            DEPTHS.into_iter().flat_map(move |depth| {
                HIDDENS.into_iter().map(move |hidden| Config { width, depth, hidden })
            })
        })
    }
    /// The parameters' `[rows, cols]` in order: first layer, hidden layers,
    /// value head, linear path.
    pub fn shapes(self) -> Vec<[usize; 2]> {
        let mut shapes = vec![[self.width, INPUT]];
        let mut last = self.width;
        for _ in 1..self.depth {
            shapes.push([self.hidden, last]);
            last = self.hidden;
        }
        shapes.push([1, last]);
        shapes.push([1, INPUT]);
        shapes
    }
    pub fn sizes(self) -> Vec<usize> {
        self.shapes().into_iter().map(|[rows, cols]| rows * cols).collect()
    }
    pub fn parameter_count(self) -> usize {
        self.sizes().into_iter().sum()
    }
}
/// Diagnostic: vary the initial weights (ROOKIE_SEED natively).
fn init_seed() -> u32 {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(v) = std::env::var("ROOKIE_SEED").ok().and_then(|v| v.parse::<u32>().ok()) {
        return v.wrapping_mul(0x9e3779b9);
    }
    0
}
/// The parameters (in `Config::shapes` order) and the raw value of each row of `x`.
pub fn build(device: &Device, x: &Tensor<2, f32>, config: Config) -> Result<(Vec<Dyn>, Tensor<2, f32>)> {
    config.validate()?;
    let mut seed = 0x726f6f6b ^ init_seed();
    let mut parameters = Vec::new();
    let mut matrix = |rows: usize, cols: usize, scale: f32| {
        let values: Vec<f32> = (0..rows * cols)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed as f64 / u32::MAX as f64 * 2. - 1.) as f32 * scale
            })
            .collect();
        let tensor = Tensor::from_slice(device, [rows, cols], &values);
        parameters.push(tensor.as_dyn().clone());
        tensor
    };
    // A wide first layer feeding narrow hidden layers, so evaluation stays cheap.
    let w = matrix(config.width, INPUT, (6. / INPUT as f32).sqrt());
    let mut h = x.matmul_t(&w).relu();
    let mut width = config.width;
    for _ in 1..config.depth {
        let w = matrix(config.hidden, width, (3. / width as f32).sqrt());
        let next = h.matmul_t(&w);
        // Equal widths form a residual layer.
        h = if width == config.hidden {
            next.add(&h).mul_scalar(std::f32::consts::FRAC_1_SQRT_2).relu()
        } else {
            next.relu()
        };
        width = config.hidden;
    }
    let value = h.matmul_t(&matrix(1, width, 0.001));
    // A linear path from the inputs straight to the value: a table of one weight
    // per piece-square learns from few games, the network above refines it.
    let value = value.add(&x.matmul_t(&matrix(1, INPUT, 0.)));
    Ok((parameters, value))
}
