use cc;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let csrc_path = PathBuf::from("csrc/model");
    let header_path = csrc_path.join("include/phi_all.h");
    let libdir_path = csrc_path.join("build");

    println!("cargo:rerun-if-changed={}", header_path.to_str().unwrap());

    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        // .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let content = [
        "#![allow(non_upper_case_globals)]",
        "#![allow(non_camel_case_types)]",
        "#![allow(non_snake_case)]",
        "",
        &bindings.to_string(),
    ]
    .join("\n");

    let out_path = PathBuf::from("src");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(out_path.join("bindings.rs"))?;

    file.write_all(content.as_bytes())?;

    // Building and copying results
    let mut cmd = Command::new("make");
    cmd.current_dir(csrc_path);
    cmd.args(["lib"]);
    let output = cmd.output().expect("failed in building");

    // Check build is valid
    if !output.status.success() {
        let stdout = String::from_utf8(output.stdout)?;
        let stderr = String::from_utf8(output.stderr)?;
        println!("stdout: {stdout}\nstderr: {stderr}");
    }

    // Location of shared library
    println!("cargo:rustc-link-search={}", libdir_path.to_str().unwrap());

    // Link library
    println!("cargo:rustc-link-lib=phi");

    tonic_build::compile_protos("proto/ll_server.proto").unwrap();
    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .flag("-cudart=shared")
        .flag("-ccbin=gcc")
        .files(&["./csrc/matmul_cuda.cu"])
        .compile("matmul_cuda");
    println!("cargo:rerun-if-changed=csrc");
    Ok(())
}
