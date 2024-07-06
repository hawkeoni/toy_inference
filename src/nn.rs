pub mod nn {

    use rayon::prelude::*;

    pub struct LinearLayer {
        w: Vec<f32>,
    }

    impl LinearLayer {
        pub fn new(val: f32, size: usize) -> Self {
            println!("Hello world");
            let w = vec![val; size];
            return Self { w };
        }

        pub fn size(&self) -> usize {
            return self.w.len();
        }

        pub fn from_vec(w: Vec<f32>) -> Self {
            return Self { w };
        }
        pub fn forward(&self, x: Vec<f32>) -> f32 {
            let mut result: f32 = 0.;
            if self.w.len() != x.len() {
                panic!("Size mismatch for w {} and x {}", self.w.len(), x.len());
            }
            let zip_iter = self.w.iter().zip(x.iter());
            for pair in zip_iter {
                result += pair.0 * pair.1;
            }
            return result;
        }

        pub fn forward_par(&self, x: Vec<f32>) -> f32 {
            if self.w.len() != x.len() {
                panic!("Size mismatch for w {} and x {}", self.w.len(), x.len());
            }
            self.w.par_iter().zip(x.par_iter())
            .map(|(w, x)| w * x)
            .sum()
            // let zip_iter = self.w.iter().zip(x.iter());
            // let result = zip_iter
            // .par_bridge()
            // .map(|x| x.0 * x.1)
            // .reduce_with(|a, b| a + b);
            // // .reduce(|| 0.0, |a, b| a + b);
            // return result.unwrap();
        }
    }
    
}