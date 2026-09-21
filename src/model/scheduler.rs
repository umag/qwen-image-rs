//! `FlowMatchEulerDiscreteScheduler` with dynamic resolution shift. **Phase 4.**
//!
//! sigmas start as `linspace(1, 1/N, N)`, get exponentially time-shifted by
//! `mu` (derived from the image sequence length), stretched so the terminal
//! sigma is `shift_terminal`, then a trailing 0 is appended. The Euler step is
//! `x_{t+1} = x_t + (sigma_next - sigma) * velocity`.

/// Scheduler config for Qwen-Image-2.1 (from scheduler_config.json).
#[derive(Debug, Clone)]
pub struct FlowConfig {
    pub base_image_seq_len: f64,
    pub max_image_seq_len: f64,
    pub base_shift: f64,
    pub max_shift: f64,
    pub shift_terminal: f64,
    pub num_train_timesteps: f64,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            base_image_seq_len: 256.0,
            max_image_seq_len: 8192.0,
            base_shift: 0.5,
            max_shift: 0.9,
            shift_terminal: 0.02,
            num_train_timesteps: 1000.0,
        }
    }
}

pub struct FlowMatchEuler {
    /// N+1 sigmas (trailing 0).
    sigmas: Vec<f64>,
    /// N timesteps in [0, 1000] (== sigma * num_train_timesteps).
    timesteps: Vec<f64>,
}

impl FlowMatchEuler {
    /// Resolution-dependent shift: `mu = image_seq_len * m + b`.
    fn calc_shift(cfg: &FlowConfig, image_seq_len: usize) -> f64 {
        let m = (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len - cfg.base_image_seq_len);
        let b = cfg.base_shift - cfg.base_image_seq_len * m;
        image_seq_len as f64 * m + b
    }

    /// Build the schedule for `num_steps` at the given image sequence length.
    pub fn new(cfg: &FlowConfig, num_steps: usize, image_seq_len: usize) -> Self {
        let mu = Self::calc_shift(cfg, image_seq_len);
        let exp_mu = mu.exp();
        // linspace(1.0, 1/N, N)
        let n = num_steps as f64;
        let mut sigmas: Vec<f64> = (0..num_steps)
            .map(|i| {
                let t = 1.0 - (i as f64) * (1.0 - 1.0 / n) / (n - 1.0);
                // exponential time shift: exp(mu) / (exp(mu) + (1/t - 1))
                exp_mu / (exp_mu + (1.0 / t - 1.0))
            })
            .collect();
        // stretch so the last sigma == shift_terminal
        let last = *sigmas.last().unwrap();
        let scale = (1.0 - last) / (1.0 - cfg.shift_terminal);
        for s in sigmas.iter_mut() {
            *s = 1.0 - (1.0 - *s) / scale;
        }
        let timesteps: Vec<f64> = sigmas.iter().map(|s| s * cfg.num_train_timesteps).collect();
        sigmas.push(0.0);
        Self { sigmas, timesteps }
    }

    /// Timestep values (feed `t / num_train_timesteps` to the DiT).
    pub fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    /// Euler step delta at index `i`: `sigma_next - sigma`.
    pub fn dt(&self, i: usize) -> f64 {
        self.sigmas[i + 1] - self.sigmas[i]
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_monotonic_and_terminal() {
        let s = FlowMatchEuler::new(&FlowConfig::default(), 40, 4096);
        assert_eq!(s.num_steps(), 40);
        // sigmas strictly decreasing to ~shift_terminal, then 0.
        for i in 0..40 {
            assert!(
                s.sigmas[i] > s.sigmas[i + 1] - 1e-9,
                "not decreasing at {i}"
            );
        }
        let terminal = s.sigmas[39];
        assert!((terminal - 0.02).abs() < 1e-6, "terminal={terminal}");
        assert_eq!(s.sigmas[40], 0.0);
        // dt is negative (denoising toward 0).
        assert!(s.dt(0) < 0.0);
    }
}
