//! Links NVIDIA NGX when its SDK is present locally.
//!
//! The SDKs are proprietary and never part of this repository; when none is
//! found the build succeeds without the NGX backends. When found,
//! `cfg(ngx)` enables them.
//!
//! * Windows: the RTX Video SDK, looked up in `$CAPTUREPLAY_RTX_VIDEO_SDK`,
//!   `$NV_RTX_VIDEO_SDK`, then `runtime/rtx-video-sdk` at the workspace root
//!   (git-ignored); `nvsdk_ngx_d.lib` (dynamic CRT, matching Rust's MSVC
//!   default) is linked.
//! * Linux: the static NGX loader of NVIDIA's DLSS SDK
//!   (`lib/Linux_x86_64/libnvsdk_ngx.a`, C linkage over libstdc++), looked up
//!   in `$CAPTUREPLAY_NGX_SDK` (the flake dev shell sets it), then
//!   `runtime/ngx-sdk`.

use std::path::PathBuf;

fn find(vars: &[&str], fallback: &str, file: &str) -> Option<PathBuf> {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    vars.iter()
        .filter_map(|v| std::env::var_os(v).map(PathBuf::from))
        .chain(std::iter::once(
            manifest.join("../../runtime").join(fallback),
        ))
        .find(|d| d.join(file).is_file())
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(ngx)");
    for v in [
        "CAPTUREPLAY_RTX_VIDEO_SDK",
        "NV_RTX_VIDEO_SDK",
        "CAPTUREPLAY_NGX_SDK",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let x64 = std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64");
    match os.as_str() {
        "windows" if x64 => {
            let lib = "lib/Windows/x64/nvsdk_ngx_d.lib";
            println!("cargo:rerun-if-changed=../../runtime/rtx-video-sdk/{lib}");
            match find(
                &["CAPTUREPLAY_RTX_VIDEO_SDK", "NV_RTX_VIDEO_SDK"],
                "rtx-video-sdk",
                lib,
            ) {
                Some(dir) => {
                    let lib = dir.join("lib/Windows/x64");
                    println!("cargo:rustc-link-search=native={}", lib.display());
                    println!("cargo:rustc-link-lib=static=nvsdk_ngx_d");
                    // Registry lookups inside the NGX loader.
                    println!("cargo:rustc-link-lib=dylib=advapi32");
                    println!("cargo:rustc-cfg=ngx");
                }
                None => println!(
                    "cargo:warning=RTX Video SDK not found: building without NGX VSR (see runtime/README.md)"
                ),
            }
        }
        "linux" if x64 => {
            let lib = "lib/Linux_x86_64/libnvsdk_ngx.a";
            println!("cargo:rerun-if-changed=../../runtime/ngx-sdk/{lib}");
            match find(&["CAPTUREPLAY_NGX_SDK"], "ngx-sdk", lib) {
                Some(dir) => {
                    let lib = dir.join("lib/Linux_x86_64");
                    println!("cargo:rustc-link-search=native={}", lib.display());
                    println!("cargo:rustc-link-lib=static=nvsdk_ngx");
                    println!("cargo:rustc-link-lib=dylib=stdc++");
                    println!("cargo:rustc-link-lib=dylib=dl");
                    println!("cargo:rustc-cfg=ngx");
                }
                None => println!(
                    "cargo:warning=NGX SDK not found: building without NGX VSR (see runtime/README.md)"
                ),
            }
        }
        _ => {}
    }
}
