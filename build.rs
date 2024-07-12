use cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .flag("-cudart=shared")
        .flag("-ccbin=gcc")
        .files(&["./csrc/matmul_cuda.cu"])
        .compile("matmul_cuda");
    println!("cargo:rerun-if-changed=csrc");
}