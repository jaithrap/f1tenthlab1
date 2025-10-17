
# Table of Contents

1.  [Addressing Limitations, Non-Native Tools&rsquo; Purpose](#orge82b27e)
    1.  [Steam Runtime](#orge5dc3d0)
    2.  [Distrobox](#orgcaf7a4f)
    3.  [Nix Flakes](#orgc0dd99f)
2.  [Installation and Usage](#orgbab3b46)
    1.  [AutoDrive](#orgf1c81e4)
    2.  [Running The F1TENTH Example](#orge8e2ef6)
    3.  [Direnv, Nix-Direnv Setup](#org0b98f61)

<details>
  <summary>hi</summary>
  Okay wow look at who&rsquo;s here using a non-FHS GNU/Linux distribution tsk tsk tsk.
</details>

<a id="orge82b27e"></a>

# Addressing Limitations, Non-Native Tools&rsquo; Purpose

If you&rsquo;re willing to give making an [FHS](https://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard)-compatible environment to run AutoDrive simulation and [example_f1tenth.py](./example_f1tenth.py) a try, go ahead. But keeping this pain-free means using [Steam Runtime](https://github.com/NixOS/nixpkgs/blob/nixos-25.05/pkgs/by-name/st/steam/package.nix#L146-L167) and [Distrobox](https://distrobox.it/).


<a id="orge5dc3d0"></a>

## Steam Runtime

This is used to run the Unity Executable. Without Steam Runtime, the prebuilt AutoDrive Linux binary fails to run on NixOS because they expect shared libraries in standard filesystem locations (that of an FHS-compatible distribution). Steam Runtime provides a set of common libraries and an FHS-like filesystem layout, allowing AutoDrive to find dynamic libraries they expect, which are not in standard locations on NixOS. Because AutoDrive is built with Unity, we can treat it as a native Linux game which Steam Runtime can easily run.

> Could not start dynamically linked executable: ./AutoDRIVE Simulator.x86<sub>64</sub>
> NixOS cannot run dynamically linked executables intended for generic
> linux environments out of the box. For more information, see:
> <https://nix.dev/permalink/stub-ld>


<a id="orgcaf7a4f"></a>

## Distrobox

This is used to run [example_f1tenth.py](./example_f1tenth.py). Virtualization with an FHS-compatible distribution is needed for the same reason as Steam Runtime&rsquo;s rationale. Despite the simple nature of the example script, a problem with hidden dependencies can be encountered: even though the websocket works, eventlet and its event handlers do not work with no indication of the problem in traceback. You can probably get it working without distrobox with a lot of spare time on hand. But it is probably better to keep things not too complicated. 🥹


<a id="orgc0dd99f"></a>

## Nix Flakes

How cool! There is a nix [flake](./flake.nix)! It **only** serves as a sample for your development environment. It is recommended that you use direnv for easy integration with IDEs / Text Editors, or so to prevent your fingers from aching because you typed &ldquo;nix develop .&rdquo; all day. Direnv setup instructions will be shown in the last heading.


<a id="orgbab3b46"></a>

# Installation and Usage


<a id="orgf1c81e4"></a>

## AutoDrive

Install AutoDrive simulation from <https://github.com/Tinker-Twins/AutoDRIVE>.
Current installation instructions support 0.3.0.
```sh
wget https://github.com/Tinker-Twins/AutoDRIVE/releases/download/Simulator-0.3.0/AutoDRIVE_Simulator_Linux.zip
```

Download steam-run either in your NixOS config or home-manager.
```nix
# configuration.nix

# steam-run is an unfree package.
nixpkgs.config.allowUnfree = true;
environment.systemPackages = with pkgs; [
    steam-run
];
```
    
```nix
# home.nix

nixpkgs.config.allowUnfree = true;
home.packages = with pkgs; [
    steam-run
];
```

Run the Unity Executable

```sh
# Make sure to change permission before running
chmod +x ./AutoDRIVE\ Simulator.x86_64
steam-run ./AutoDRIVE\ Simulator.x86_64
```


<a id="orge8e2ef6"></a>

## Running The F1TENTH Example

First, we&rsquo;ll setup distrobox according to the [NixOS Wiki](https://wiki.nixos.org/wiki/Distrobox). In your configuration.nix file:
```nix
virtualisation.podman = {
    enable = true;
    dockerCompat = true;
};

environment.systemPackages = [ pkgs.distrobox ];
```

Second, we&rsquo;ll setup a distrobox image and then enter the container.
```sh
# Creating image
distrobox create --image ubuntu:latest --name autodrive-ubuntu
# Entering container
distrobox enter autodrive-ubuntu
```
Clone this repo for the example script.
```sh
git clone https://github.com/uci-f1tenth/UCI_AutoDRIVE_labs
cd UCI_AutoDRIVE_labs
```

Install uv and run [example_f1tenth.py](./example_f1tenth.py)

```sh
sudo curl -LsSf https://astral.sh/uv/install.sh | sh
uv run example_f1tenth.py
```

<a id="org0b98f61"></a>

## Direnv, Nix-Direnv Setup

The official page for this setup is here: <https://github.com/nix-community/nix-direnv>.
First install direnv on NixOS either in configuration.nix or home.nix.
Configuration differs according to what shell you&rsquo;re using: <https://search.nixos.org/options?query=programs.direnv>.

```nix
programs.direnv = {
    enable = true;
    # I'm using zsh. Choose the shell you're using. Make sure your shell is enabled as well.
    enableZshIntegration = true;
    nix-direnv.enable = true;
    nix-direnv.package = pkgs-unstable.nix-direnv;
};
```

Hook your shell: <https://direnv.net/docs/hook.html>.
This is how I did mine via home manager. The fields should be the same for a NixOS module or configuration file as well.
```nix
programs.zsh = {
    enable = true;
    initExtra = ''
    eval "$(direnv hook zsh)"
    '';
};
```

Make sure you&rsquo;re in the same directory [flake.nix](./flake.nix) is at. Copy [.envrc-nixos](./.envrc-nixos) to .envrc.
```sh
cp .envrc-nixos .envrc
```

Then allow direnv there.
```sh
direnv allow
```
