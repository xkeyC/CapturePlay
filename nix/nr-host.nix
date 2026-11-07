# captureplay-nr-host.exe + captureplay_nvngx.dll: the Windows helper that
# runs DLSS Neural Rendering under Proton for the Linux build. Call it with
# a mingw cross package set (`pkgsCross.mingwW64.callPackage`). NVIDIA's
# nvngx_dlssnr.dll is not included; CapturePlay looks for it in
# runtime/nvidia-nr, $CAPTUREPLAY_NR_RUNTIME or DLSS5VKLayer's import folder.
{ lib, rustPlatform, windows }:

rustPlatform.buildRustPackage {
  pname = "captureplay-nr-host";
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
  cargoBuildFlags = [ "-p" "nr-host" "-p" "nr-bridge" ];
  doCheck = false;
  # Rust's windows-gnu std links winpthreads statically.
  buildInputs = [ windows.pthreads ];

  # The bridge is a cdylib, which the install hook does not pick up.
  postInstall = ''
    dll=$(find target -path '*/release/captureplay_nvngx.dll' | head -1)
    install -Dm644 "$dll" $out/bin/captureplay_nvngx.dll
  '';

  meta = {
    description = "Proton-hosted DLSS Neural Rendering helper for CapturePlay";
    homepage = "https://github.com/xkeyC/CapturePlay";
    license = lib.licenses.gpl3Plus;
  };
}
