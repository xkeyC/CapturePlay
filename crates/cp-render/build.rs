//! Compiles HLSL to signed DXIL with the Windows SDK's `dxc.exe`.
//!
//! Override the compiler with `DXC_PATH`. Shaders are embedded with
//! `include_bytes!` from `OUT_DIR`, so the binary has no runtime compiler
//! dependency.

use std::path::{Path, PathBuf};
use std::process::Command;

struct Shader {
    file: &'static str,
    entry: &'static str,
    profile: &'static str,
    out: &'static str,
    defines: &'static [&'static str],
}

const SHADERS: &[Shader] = &[
    Shader {
        file: "convert.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "convert_planar",
        defines: &["PLANAR_420=1"],
    },
    Shader {
        file: "convert.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "convert_yuyv",
        defines: &["PACKED_YUYV=1"],
    },
    Shader {
        file: "convert.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "convert_uyvy",
        defines: &["PACKED_UYVY=1"],
    },
    Shader {
        file: "convert.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "convert_bgra",
        defines: &["PACKED_BGRA=1"],
    },
    Shader {
        file: "output.hlsl",
        entry: "vs_main",
        profile: "vs_6_0",
        out: "output_vs",
        defines: &[],
    },
    Shader {
        file: "output.hlsl",
        entry: "ps_main",
        profile: "ps_6_0",
        out: "output_ps",
        defines: &[],
    },
    Shader {
        file: "egui.hlsl",
        entry: "vs_main",
        profile: "vs_6_0",
        out: "egui_vs",
        defines: &[],
    },
    Shader {
        file: "egui.hlsl",
        entry: "ps_main",
        profile: "ps_6_0",
        out: "egui_ps",
        defines: &[],
    },
    Shader {
        file: "vfg.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "vfg_encode",
        defines: &["ENCODE=1"],
    },
    Shader {
        file: "vfg.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "vfg_decode",
        defines: &["ENCODE=0"],
    },
    Shader {
        file: "vfg.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "ngx_encode",
        defines: &["ENCODE=1", "TEXTURE_IO=1"],
    },
    Shader {
        file: "vfg.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "ngx_decode",
        defines: &["ENCODE=0", "TEXTURE_IO=1"],
    },
    Shader {
        file: "nr.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "nr_encode",
        defines: &["ENCODE=1"],
    },
    Shader {
        file: "nr.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "nr_composite",
        defines: &[],
    },
    Shader {
        file: "nr.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "nr_fill_depth",
        defines: &["FILL_DEPTH=1"],
    },
    Shader {
        file: "nr.hlsl",
        entry: "main",
        profile: "cs_6_0",
        out: "nr_fill_motion",
        defines: &["FILL_MOTION=1"],
    },
];

fn find_dxc() -> PathBuf {
    if let Ok(p) = std::env::var("DXC_PATH") {
        return PathBuf::from(p);
    }
    let base = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(base)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.join(r"x64\dxc.exe").exists())
                .collect()
        })
        .unwrap_or_default();
    versions.sort();
    versions
        .pop()
        .map(|p| p.join(r"x64\dxc.exe"))
        .unwrap_or_else(|| PathBuf::from("dxc.exe"))
}

fn main() {
    let dxc = find_dxc();
    let src_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir")).join("shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("out dir"));
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=DXC_PATH");

    for s in SHADERS {
        let out = out_dir.join(format!("{}.cso", s.out));
        let mut cmd = Command::new(&dxc);
        cmd.arg("-nologo")
            .args(["-T", s.profile, "-E", s.entry])
            .args(["-O3", "-Zpr", "-HV", "2021"])
            .arg("-I")
            .arg(&src_dir)
            .arg("-Fo")
            .arg(&out)
            .arg(src_dir.join(s.file));
        for d in s.defines {
            cmd.arg("-D").arg(d);
        }
        let result = cmd
            .output()
            .unwrap_or_else(|e| panic!("cannot run {}: {e} (set DXC_PATH)", dxc.display()));
        if !result.status.success() {
            panic!(
                "dxc failed for {} ({}):\n{}{}",
                s.file,
                s.out,
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}
