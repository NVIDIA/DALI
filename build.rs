use std::env;
use std::path::PathBuf;

fn main() {

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=/opt/nvidia/cvcuda0/lib/");

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo:rustc-link-lib=dali");
    println!("cargo:rustc-link-lib=dali_core");
    println!("cargo:rustc-link-lib=dali_kernnels");
    println!("cargo:rustc-link-lib=dali_operators");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .clang_arg("-Iinclude/")
        .header("include/dali/c_api.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}