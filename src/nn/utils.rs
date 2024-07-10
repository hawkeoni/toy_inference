use rand::Rng;

pub fn create_random_vector(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let res: Vec<f32> = (0..size)
    .map(|_| rng.gen::<f32>())
    .collect();
    return res;
}

pub fn create_random_matrix(dim0: usize, dim1: usize) -> Vec<Vec<f32>>{
    let res: Vec<Vec<f32>> = (0..dim0)
    .map(|_| create_random_vector(dim1))
    .collect();
    return res;
}