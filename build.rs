use cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .flag("-cudart=shared")
        .files(&["./csrc/matmul_cuda.cu"])
        .compile("matmul_cuda");
    println!("cargo:rerun-if-changed=csrc");

    tonic_build::compile_protos("proto/ll_server.proto").unwrap();
}