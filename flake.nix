{
  description = "CapturePlay development shell (Linux: Vulkan + Wayland + V4L2)";

  # Same nixpkgs revision as the development machine's NixOS.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/02c087f3fe673c9e317f206a627cb6b910336dbe";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      # NVIDIA's static NGX loader for Linux (DLSS SDK, NVIDIA RTX SDKs
      # License; never part of this repository): enables VSR on Vulkan.
      ngxRev = "374959484e79a640feaba44c93ac8cfb0a03f5b5";
      ngxSdk = pkgs.runCommand "nvidia-ngx-sdk-linux" { } ''
        install -Dm644 ${
          pkgs.fetchurl {
            url = "https://raw.githubusercontent.com/NVIDIA/DLSS/${ngxRev}/lib/Linux_x86_64/libnvsdk_ngx.a";
            sha256 = "3321c4ca9adf71f345550d10bf2148ed38003151b03ec5f81c99d00949ab8202";
          }
        } $out/lib/Linux_x86_64/libnvsdk_ngx.a
      '';
      # dlopen()ed at run time: the Vulkan loader, winit's Wayland/X11 backends.
      runtimeLibs = with pkgs; [
        vulkan-loader
        wayland
        libxkbcommon
        libx11
        libxcursor
        libxi
        libxrandr
      ];
    in
    {
      # The Rust toolchain comes from rustup (rust-version in Cargo.toml).
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          directx-shader-compiler # HLSL → SPIR-V (crates/cp-render/build.rs)
          pkg-config
          rustPlatform.bindgenHook # libclang + headers for pipewire-sys
          spirv-tools
          vulkan-tools
          vulkan-validation-layers
          v4l-utils
        ];
        # Linked by cp-audio (pkg-config finds the .dev output here).
        buildInputs = [ pkgs.pipewire ];
        CAPTUREPLAY_NGX_SDK = ngxSdk;
        DXC_PATH = "${pkgs.directx-shader-compiler}/bin/dxc";
        # /run/opengl-driver/lib: the NVIDIA driver's libcuda / NGX for VFG and VSR.
        LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath runtimeLibs}:/run/opengl-driver/lib";
        # Enabled with CAPTUREPLAY_VK_DEBUG=1.
        VK_ADD_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
      };
    };
}
