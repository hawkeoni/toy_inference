pub mod utils;

use rayon::prelude::*;
use utils::{create_random_matrix, create_random_vector};

pub enum InferenceMode {
    Naive,
    Rayon,
    Gpu
}


pub struct LinearLayer {
    w: Vec<Vec<f32>>,
    mode: InferenceMode,
    fan_in: usize,
    fan_out: usize,
}

impl LinearLayer {

    pub fn size(&self) -> (usize, usize) {
        return (self.w.len(), self.w[0].len());
    }

    pub fn new(w: Vec<Vec<f32>>, mode: InferenceMode) -> Self {
        let fan_in = w.len();
        let fan_out = w[0].len();
        return Self { 
            w,
            mode,
            fan_in,
            fan_out,
        };
    }

    pub fn forward(&self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>>{
        // self - fan_in, fan_out
        // x - batch, fan_in
        // output - batch, fan_out
        // output = xA; output_ij = \Sum_k x_ik * A_kj
        if self.fan_in != x[0].len() {
            panic!("Size mismatch for w {} and x {}", self.fan_in, x[1].len());
        }
        match self.mode {
            InferenceMode::Naive => return self.forward_cpu(x),
            InferenceMode::Rayon => return self.forward_rayon(x),
            InferenceMode::Gpu => return self.forward_gpu(x),
        }
    }

    fn forward_cpu(&self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let batch_size = x.len();
        let mut result: Vec<Vec<f32>> = vec![vec![0.0; self.fan_out]; batch_size];
        for i in 0..batch_size {
            for j in 0..self.fan_out {
                for k in 0..self.fan_in {
                    result[i][j] += x[i][k] * self.w[k][j];
                }
            }
        }
        return result;
    }

    fn forward_rayon(&self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let batch_size = x.len();
        let mut result: Vec<Vec<f32>> = vec![vec![0.0; self.fan_out]; batch_size];
        
        result.par_iter_mut().enumerate().for_each(|(i, rrow)| {
            for j in 0..self.fan_out {
                for k in 0..self.fan_in {
                    rrow[j] += x[i][k] * self.w[k][j];
                }
            }
        });

        return result;
    }

    fn forward_gpu(&self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        return self.forward_cpu(x);
    }
    
}

#[cfg(test)]
mod tests {

    use super::{utils::create_random_matrix, InferenceMode, LinearLayer};

    #[test]
    fn rayon_check() {
        let batch_size = 3;
        let dim0 = 5; let dim1 = 7;
        let x = create_random_matrix(batch_size, dim0);
        let w = create_random_matrix(dim0, dim1);
        let layer_naive = LinearLayer::new(w.clone(), InferenceMode::Naive);
        let layer_rayon = LinearLayer::new(w.clone(), InferenceMode::Rayon);

        let res_naive = layer_naive.forward(&x);
        let res_rayon = layer_rayon.forward(&x);

        assert_eq!(res_naive.len(), res_rayon.len());
        assert_eq!(res_naive[0].len(), res_rayon[0].len());

        for i in 0..res_naive.len() {
            for j in 0..res_naive[0].len() {
                assert_eq!(res_naive[i][j], res_rayon[i][j]);
            }
        }

        

        return;
    }

}
