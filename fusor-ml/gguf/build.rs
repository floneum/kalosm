fn main() {
    // Tell Cargo to re-run this build script if it changes
    println!("cargo:rerun-if-changed=build.rs");

    // Register the nightly cfg for check-cfg to avoid warnings
    println!("cargo::rustc-check-cfg=cfg(nightly)");

    // Set a cfg flag if we're on nightly
    #[rustversion::nightly]
    fn set_nightly_cfg() {
        println!("cargo:rustc-cfg=nightly");
    }

    #[rustversion::not(nightly)]
    fn set_nightly_cfg() {}

    set_nightly_cfg();
}
