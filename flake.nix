{
  description = "CapturePlay development shell (Linux: Vulkan + Wayland + V4L2)";

  # Same nixpkgs revision as the development machine's NixOS.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/02c087f3fe673c9e317f206a627cb6b910336dbe";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
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
          spirv-tools
          vulkan-tools
          vulkan-validation-layers
          v4l-utils
        ];
        DXC_PATH = "${pkgs.directx-shader-compiler}/bin/dxc";
        # /run/opengl-driver/lib: the NVIDIA driver's libcuda / NGX for VFG and VSR.
        LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath runtimeLibs}:/run/opengl-driver/lib";
        # Enabled with CAPTUREPLAY_VK_DEBUG=1.
        VK_ADD_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
      };
    };
}
