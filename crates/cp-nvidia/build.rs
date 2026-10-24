//! Links the NVIDIA RTX Video SDK (NGX) when it is present locally.
//!
//! The SDK is proprietary and never part of this repository. It is looked up
//! in `$CAPTUREPLAY_RTX_VIDEO_SDK`, `$NV_RTX_VIDEO_SDK`, then
//! `runtime/rtx-video-sdk` at the workspace root (git-ignored). When found,
//! `nvsdk_ngx_d.lib` (dynamic CRT, matching Rust's MSVC default) is linked
//! and `cfg(ngx)` enables the NGX backend; otherwise the build succeeds
//! without it.

use std::path::PathBuf;

fn sdk_dir() -> Option<PathBuf> {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    ["CAPTUREPLAY_RTX_VIDEO_SDK", "NV_RTX_VIDEO_SDK"]
        .iter()
        .filter_map(|v| std::env::var_os(v).map(PathBuf::from))
        .chain(std::iter::once(
            manifest.join("../../runtime/rtx-video-sdk"),
        ))
        .find(|d| d.join("lib/Windows/x64/nvsdk_ngx_d.lib").is_file())
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(ngx)");
    println!("cargo:rerun-if-env-changed=CAPTUREPLAY_RTX_VIDEO_SDK");
    println!("cargo:rerun-if-env-changed=NV_RTX_VIDEO_SDK");
    println!("cargo:rerun-if-changed=../../runtime/rtx-video-sdk/lib/Windows/x64/nvsdk_ngx_d.lib");
    let target_ok = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64");
    match sdk_dir().filter(|_| target_ok) {
        Some(dir) => {
            let lib = dir.join("lib/Windows/x64");
            println!("cargo:rustc-link-search=native={}", lib.display());
            println!("cargo:rustc-link-lib=static=nvsdk_ngx_d");
            // Registry lookups inside the NGX loader.
            println!("cargo:rustc-link-lib=dylib=advapi32");
            println!("cargo:rustc-cfg=ngx");
        }
        None => {
            println!(
                "cargo:warning=RTX Video SDK not found: building without NGX VSR (see runtime/README.md)"
            );
        }
    }
}
