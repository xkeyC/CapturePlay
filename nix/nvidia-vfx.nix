# NVIDIA Video Effects runtime for Linux (the `nvidia-vfx` wheel): frame
# interpolation (libnvVFXVideoFrameGeneration.so, through CUDA) and the RTX
# VSR snippet (libnvidia-ngx-vsr.so, through NGX). Unfree; kept exactly as
# shipped (no fixup: NGX checks the snippets' signatures, and the libraries
# find each other through their own RUNPATH and CapturePlay's preloading).
# Same wheel as chaos-hdr's, so both share one download.
{ lib, stdenvNoCC, fetchurl, unzip }:

stdenvNoCC.mkDerivation {
  pname = "nvidia-vfx-runtime";
  version = "0.2.0.0";
  src = fetchurl {
    url = "https://pypi.nvidia.com/nvidia-vfx/nvidia_vfx-0.2.0.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    sha256 = "0252866a7b798a6f92c8543443d88f716e8c5bbc6a7ddd77a2754728dada9fa8";
  };
  nativeBuildInputs = [ unzip ];
  unpackPhase = "unzip -q $src 'nvvfx/libs/*' 'nvidia_vfx.libs/*' 'nvidia_vfx-0.2.0.0.dist-info/*'";
  installPhase = ''
    mkdir -p $out
    cp -r nvvfx nvidia_vfx.libs nvidia_vfx-0.2.0.0.dist-info $out/
  '';
  dontFixup = true;
  meta = {
    description = "NVIDIA Video Effects SDK runtime (frame generation, RTX VSR)";
    homepage = "https://pypi.nvidia.com/nvidia-vfx/";
    license = lib.licenses.unfree;
    platforms = [ "x86_64-linux" ];
  };
}
