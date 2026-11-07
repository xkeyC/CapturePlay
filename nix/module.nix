# programs.capture-play: installs CapturePlay, optionally with NVIDIA's
# Video Effects runtime (frame interpolation, RTX VSR) and the Proton
# helper for DLSS NR.
{ config, lib, pkgs, ... }:

let
  cfg = config.programs.capture-play;
in
{
  options.programs.capture-play = {
    enable = lib.mkEnableOption "CapturePlay, the capture card player";
    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.callPackage ./package.nix { };
      defaultText = lib.literalExpression "pkgs.callPackage ./package.nix { }";
      description = "CapturePlay package, before the NVIDIA runtimes below are wired in.";
    };
    nvidiaVfx = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Ship NVIDIA's Video Effects runtime (the unfree `nvidia-vfx` wheel,
        about 600 MB) for AI frame interpolation and RTX VSR.
      '';
    };
    dlssNr = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Ship the Proton helper for DLSS Neural Rendering. Proton comes from
        Steam (through steam-run on NixOS) and NVIDIA's nvngx_dlssnr.dll is
        user-provided (see runtime/README.md).
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [
      (cfg.package.override {
        vfxRuntime = if cfg.nvidiaVfx then pkgs.callPackage ./nvidia-vfx.nix { } else null;
        nrHost = if cfg.dlssNr then pkgs.pkgsCross.mingwW64.callPackage ./nr-host.nix { } else null;
      })
    ];
  };
}
