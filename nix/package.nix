# CapturePlay for Linux (Vulkan + Wayland). The NVIDIA pieces are optional
# and not part of the default package: pass `vfxRuntime` (nvidia-vfx.nix:
# frame interpolation and RTX VSR) and `nrHost` (nr-host.nix: DLSS NR under
# Proton; the model DLL itself is user-provided) to wire them in.
{ lib
, rustPlatform
, fetchurl
, runCommand
, makeWrapper
, makeDesktopItem
, copyDesktopItems
, directx-shader-compiler
, pkg-config
, pipewire
, vulkan-loader
, wayland
, libxkbcommon
, libx11
, libxcursor
, libxi
, libxrandr
, vfxRuntime ? null
, nrHost ? null
}:

let
  # NVIDIA's static NGX loader for Linux (DLSS SDK, NVIDIA RTX SDKs License),
  # linked for RTX VSR on Vulkan. Same file as in the dev shell.
  ngxRev = "374959484e79a640feaba44c93ac8cfb0a03f5b5";
  ngxSdk = runCommand "nvidia-ngx-sdk-linux" { } ''
    install -Dm644 ${
      fetchurl {
        url = "https://raw.githubusercontent.com/NVIDIA/DLSS/${ngxRev}/lib/Linux_x86_64/libnvsdk_ngx.a";
        sha256 = "3321c4ca9adf71f345550d10bf2148ed38003151b03ec5f81c99d00949ab8202";
      }
    } $out/lib/Linux_x86_64/libnvsdk_ngx.a
  '';
  # dlopen()ed at run time: the Vulkan loader, winit's Wayland/X11 backends.
  runtimeLibs = [
    vulkan-loader
    wayland
    libxkbcommon
    libx11
    libxcursor
    libxi
    libxrandr
  ];
in
rustPlatform.buildRustPackage {
  pname = "capture-play";
  version = (lib.importTOML ../Cargo.toml).workspace.package.version;

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../crates
    ];
  };
  cargoLock.lockFile = ../Cargo.lock;
  cargoBuildFlags = [ "-p" "capture-play" ];
  # The GPU tests need a device; `nix develop` runs them.
  doCheck = false;

  nativeBuildInputs = [
    directx-shader-compiler
    pkg-config
    rustPlatform.bindgenHook
    makeWrapper
    copyDesktopItems
  ];
  buildInputs = [ pipewire ];

  env = {
    CAPTUREPLAY_NGX_SDK = ngxSdk;
    DXC_PATH = "${directx-shader-compiler}/bin/dxc";
  };

  desktopItems = [
    (makeDesktopItem {
      name = "capture-play";
      desktopName = "CapturePlay";
      comment = "Low-latency capture card player";
      exec = "capture-play";
      icon = "video-display";
      categories = [ "AudioVideo" "Video" "Player" ];
      # The window's Wayland app_id / X11 class (window rules match it).
      startupWMClass = "capture-play";
    })
  ];

  # /run/opengl-driver/lib: the NVIDIA driver's libcuda and NGX core.
  postFixup = ''
    wrapProgram $out/bin/capture-play \
      --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath runtimeLibs}:/run/opengl-driver/lib \
      ${lib.optionalString (vfxRuntime != null) "--set-default CAPTUREPLAY_VFX_RUNTIME ${vfxRuntime}"} \
      ${lib.optionalString (nrHost != null) "--set-default CAPTUREPLAY_NR_HOST ${nrHost}/bin/captureplay-nr-host.exe"}
  '';

  meta = {
    description = "Low-latency capture card player with NVIDIA frame interpolation, RTX VSR and HDR";
    homepage = "https://github.com/xkeyC/CapturePlay";
    license = lib.licenses.gpl3Plus;
    mainProgram = "capture-play";
    platforms = [ "x86_64-linux" ];
  };
}
