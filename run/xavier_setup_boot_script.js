#!/usr/bin/env sh
"\"",`$(echo --% ' |out-null)" >$null;function :{};function getDenoVersion{<#${/*'>/dev/null )` 2>/dev/null;getDenoVersion() { #>
echo "1.41.3";: --% ' |out-null <#';};DENO_INSTALL="$HOME/.deno/$(getDenoVersion)";deno_version="v$(getDenoVersion)";deno="$DENO_INSTALL/bin/deno";target_script="$0";disable_url_run="";if [ -n "$url_" ] && [ -z "$disable_url_run" ];then if [ "${url_#http}" = "$url_" ];then url_="https://$url_";fi;target_script="$url_";fi;if [ -x "$deno" ];then exec "$deno" run -q -A --no-lock --no-config "$target_script" "$@";elif [ -f "$deno" ];then chmod +x "$deno" && exec "$deno" run -q -A --no-lock --no-config "$target_script" "$@";fi;has () { command -v "$1" >/dev/null;};if ! has curl;then if ! has wget;then curl () { wget --output-document="$5" "$6";};else echo "Sorry this script needs curl or wget, please install one or the other and re-run";exit 1;fi;fi;if [ "$(uname)" = "Darwin" ];then unzip () { /usr/bin/tar xvf "$4" -C "$2" 2>/dev/null 1>/dev/null;};fi;if ! has unzip && ! has 7z;then echo "Either the unzip or 7z command are needed for this script";echo "Should I try to install unzip for you?";read ANSWER;echo;if [ "$ANSWER" =~ ^[Yy] ];then if has nix-shell;then unzip_path="$(NIX_PATH='nixpkgs=https://github.com/NixOS/nixpkgs/archive/release-25.05.tar.gz' nix-shell -p unzip which --run "which unzip")" alias unzip="$unzip_path" else;if has apt-get;then _install="apt-get install unzip -y";elif has dnf;then _install="dnf install unzip -y";elif has pacman;then _install="pacman -S --noconfirm unzip";else echo "Sorry, I don't know how to install unzip on this system";echo "Please install unzip manually and re-run this script";exit 1;fi;if [ "$(whoami)" = "root" ];then "$_install";elif has sudo;then sudo "$_install";elif has doas;then doas "$_install";else "$_install";fi;fi;fi;if ! has unzip;then echo "";echo "So I couldn't find an 'unzip' or '7z' command";echo "And I tried to auto install unzip, but it seems that failed";echo "Please install the unzip or 7z command manually then re-run this script";exit 1;fi;fi;if ! command -v unzip >/dev/null && ! command -v 7z >/dev/null;then echo "Error: either unzip or 7z is required to install Deno (see: https://github.com/denoland/deno_install#either-unzip-or-7z-is-required )." 1>&2;exit 1;fi;if [ "$OS" = "Windows_NT" ];then target="x86_64-pc-windows-msvc";else case $(uname -sm) in "Darwin x86_64") target="x86_64-apple-darwin" ;;"Darwin arm64") target="aarch64-apple-darwin" ;;"Linux aarch64") target="aarch64-unknown-linux-gnu" ;;*) target="x86_64-unknown-linux-gnu" ;;esac fi;deno_uri="https://dl.deno.land/release/${deno_version}/deno-${target}.zip";deno_install="${DENO_INSTALL:-$HOME/.deno}";bin_dir="$deno_install/bin";exe="$bin_dir/deno";if [ ! -d "$bin_dir" ];then mkdir -p "$bin_dir";fi;curl --fail --location --progress-bar --output "$exe.zip" "$deno_uri";if command -v unzip >/dev/null;then unzip -d "$bin_dir" -o "$exe.zip";else 7z x -o"$bin_dir" -y "$exe.zip";fi;chmod +x "$exe";rm "$exe.zip";exec "$deno" run -q -A --no-lock --no-config "$target_script" "$@";#>};$DenoInstall = "${HOME}/.deno/$(getDenoVersion)";$BinDir = "$DenoInstall/bin";$DenoExe = "$BinDir/deno.exe";$TargetScript = "$PSCommandPath";$DisableUrlRun = "";if ($url_ -and -not($DisableUrlRun)) { if (-not($url -match '^http')) { $url_="https://$url_";} $TargetScript = "$url_";};if (-not(Test-Path -Path "$DenoExe" -PathType Leaf)) { $DenoZip = "$BinDir/deno.zip";$DenoUri = "https://github.com/denoland/deno/releases/download/v$(getDenoVersion)/deno-x86_64-pc-windows-msvc.zip";[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;if (!(Test-Path $BinDir)) { New-Item $BinDir -ItemType Directory | Out-Null;};curl.exe --ssl-revoke-best-effort -Lo $DenoZip $DenoUri;tar.exe xf $DenoZip -C $BinDir;Remove-Item $DenoZip;};& "$DenoExe" run -q -A --no-lock --no-config "$TargetScript" @args;Exit $LastExitCode;<# 
# */0}`;
import { OperatingSystem } from "https://deno.land/x/quickr@0.6.72/main/operating_system.js"
import { FileSystem, glob } from "https://deno.land/x/quickr@0.6.72/main/file_system.js"
import run from "https://esm.sh/jsr/@david/dax@0.42.0"
import { Console, clearAnsiStylesFrom, black, white, red, green, blue, yellow, cyan, magenta, lightBlack, lightWhite, lightRed, lightGreen, lightBlue, lightYellow, lightMagenta, lightCyan, blackBackground, whiteBackground, redBackground, greenBackground, blueBackground, yellowBackground, magentaBackground, cyanBackground, lightBlackBackground, lightRedBackground, lightGreenBackground, lightYellowBackground, lightBlueBackground, lightMagentaBackground, lightCyanBackground, lightWhiteBackground, bold, reset, dim, italic, underline, inverse, strikethrough, gray, grey, lightGray, lightGrey, grayBackground, greyBackground, lightGrayBackground, lightGreyBackground, } from "https://deno.land/x/quickr@0.6.72/main/console.js"

// 
// 
// How does this work
// 
// 
    // 1. this script sets up a systemd service
    // 2. on boot (of the xavier) that service gets triggered
    // 3. that service MUST run a shell (bash) script
    // 4. that bash script is started as ROOT with absolutely nothing setup, which is a problem because we need things like the PYTHONPATH
        // so this file generates/checks/ensures that bash script exists
            // and it makes sure all the permissions are correct
            // and it makes sure the repo/main.py didn't move
            // and it keeps all the code in one place (here) so if you edit a name, you only edit it once (here) and then run the setup script again
        // that bash script (which is the contentsOfInfiniteRunScript variable) is basically a wrapper around the actual command we care about
        // the actual command we care about (pathToOnBootCommand) is also a bash script

// helpers
const shellEscape = (arg)=>"'"+arg.replace(/'/g,`'"'"'`)+"'"
const userName = Deno.env.get("USER")
const home = Deno.env.get("HOME")

// 
// 
// parameters
// 
// 

const envVarsFile = FileSystem.makeAbsolutePath(`${home}/.zshrc`)
// note: the .ignore is because we don't want this to be synced with git
const pathToOnBootCommand          = FileSystem.makeAbsolutePath(`${FileSystem.thisFolder}/boot_command.ignore`)
const pathToActualInfinteRunScript = FileSystem.makeAbsolutePath(`${FileSystem.thisFolder}/cv_dark_infinite_run.ignore.sh`)
const pathToBootLogFile            = FileSystem.makeAbsolutePath(`${home}/boot.log`)
const pathToOldBootLogFile         = FileSystem.makeAbsolutePath(`${home}/boot.old.log`)
const projectFolder                = FileSystem.makeAbsolutePath(await FileSystem.walkUpUntil(".git/config"))
const pathToMainPy                 = `${projectFolder}/main/main.py`
const nameOfStartService           = "cv_dark_boot"
const autobootIdFile               = FileSystem.makeAbsolutePath(`${FileSystem.thisFolder}/autoboot_id`)

// 
// this initialized one time (no overwrite)
// 
const defaultBootCommand = `#!/usr/bin/env bash

# Edit me as needed
python3 ${shellEscape(pathToMainPy)} \
    @WE_BLUE \
    @BOARD=XAVIER \
    @GPU=TENSOR_RT \
    @CAMERA=REALSENSE \
    @SHOWTIME \
    autoboot_id:"$(${shellEscape(autobootIdFile)})"
`

// 
// this overwrites every time setup is run
// 
let pathToZsh
try {
    pathToZsh = await run`which zsh`.text()
} catch (error) {
    throw Error(`\n\nI couldn't find zsh. Previously the script had a hard time running using bash instead of zsh\n\n`)
}
const contentsOfInfiniteRunScript = `#!/usr/bin/env bash

# 
# 
#  !!! THIS IS A GENERATED FILE !!! ${
    // 
    // Note: if you're reading this, ignore the "EDIT THE REAL ONE"
    //       (this is the "real one")
    // 
''}
#  EDIT THE REAL ONE INSIDE OF ${FileSystem.thisFile.replace(/\n/g,"")}
#  then run that file
# 

# note: when the boot script runs NOTHING is setup
#       this script:
#       1. sets up all the normal user stuff
#       2. logs all output to $HOME/boot.log
#       3. runs the following IN A LOOP: ${pathToOnBootCommand.replace(/\n/g,"")}

this_username=${shellEscape(userName)}
this_home=${shellEscape(home)}
boot_log=${shellEscape(pathToBootLogFile)}
old_boot_log=${shellEscape(pathToOldBootLogFile)}

# handle logging file
rm -f "$old_boot_log"
# copy old file
if [ -f "$boot_log" ]
then
    cp "$boot_log" "$old_boot_log"
fi
rm -f "$boot_log"

export HOME="$this_home"
sudo -u "$this_username" -E this_username="$this_username" this_home="$this_home" -- ${shellEscape(pathToZsh)} -c ${shellEscape(`
    echo "#"
    echo "#"
    echo "#"
    echo "# BOOTING UP XAVIER"
    echo "#"
    echo "#"
    echo "#"
    
    while true
    do
        . ${shellEscape(envVarsFile)}
        cd ${shellEscape(projectFolder)}
        . ${shellEscape(pathToOnBootCommand)}
        
        echo "#"
        echo "#"
        echo "#"
        echo "# PROCESS DIED"
        echo "# [RESTARTING now]"
        echo "#"
        echo "#"
        echo "#"
    done
`)} 2>>"$boot_log" 1>>"$boot_log"
`

// 
// sanity checks
// 
    if (!OperatingSystem.commonChecks.isLinux || OperatingSystem.commonChecks.isWsl) {
        throw Error(`\n\nThis script is ONLY for the Jetson. DO NOT RUN IT ON YOUR PC.\n\n`)
    }
    if (Deno.env.get("FORNIX_FOLDER")) {
        throw Error(`\n\nDon't run this script inside of the project env! (commands/start)\nIt needs to know the jetsons real home folder, and the project env uses a fake home folder\n`)
    }
    if (!FileSystem.sync.info(pathToMainPy).isFile) {
        throw Error(`\n\n\nI expected the main.py file to exist here:\n    ${pathToMainPy}\nBut it doesn't.\nOpen up the setup_boot_script.js file and fix that path to point to the actual file.\n\n`)
    }
    if (!nameOfStartService.match(/^[a-zA-Z_][a-zA-Z0-9_]+$/)) {
        throw Error(`\n\nInside of ${FileSystem.thisFile}.\nYou must have changed the nameOfStartService to something with invalid characters.\nIt needs to be a valid variable name\n`)
    }
    if (home != `/home/${userName}`) {
        console.warn(`\n\nThere is something weird going on\nNormally ${cyan("$HOME")} is /home/$username (aka it would be ${`/home/${userName}`})\nBut this time ${cyan("$HOME")} was ${home}\n\nIF YOU'RE RUNNING THIS INSIDE OF THE PROJECT ENV (commands/start)\nTHATS A PROBLEM\nIf that^ home is a fake home folder, this is going to break the boot script\n\n`)
        if (!(await Console.askFor.yesNo("Do you want to continue anyway? (yes/no)"))) {
            console.log(`\n\nAight...\n`)
            Deno.exit(1)
        }
        console.log(`\n\nContinuing anyway...\n`)
    }

// 
// 
// note: everything below runs right-now (user manually running the setup command)
// the strings-of-code like contentsOfInfiniteRunScript are what is actually run during boot-up
// 
// 

    // 
    // 
    // (setup) generate files
    // 
    // 

    // generate pathToOnBootCommand if it doesn't exist
    if (!FileSystem.sync.info(pathToOnBootCommand).isFile) {
        console.log(`\n\nGenerating ${pathToOnBootCommand}\n`)
        FileSystem.sync.write({
            path: pathToOnBootCommand,
            data: defaultBootCommand,
        })
        console.log(`Note: moving/renaming that file will break the boot script`)
    }
    // ensure those scripts are executable
    await FileSystem.addPermissions({
        path: pathToOnBootCommand,
        permissions: {
            owner: {
                canExecute: true,
            },
            group: {
                canExecute: true,
            },
            other: {
                canExecute: true,
            }
        },
    })

    // generate pathToActualInfinteRunScript if it doesn't exist
    console.log(`\n\nGenerating ${pathToActualInfinteRunScript}\n`)
    FileSystem.sync.write({
        path: pathToActualInfinteRunScript,
        data: contentsOfInfiniteRunScript,
        overwrite: true,
    })
    // ensure those scripts are executable
    await FileSystem.addPermissions({
        path: pathToActualInfinteRunScript,
        permissions: {
            owner: {
                canExecute: true,
            },
            group: {
                canExecute: true,
            },
            other: {
                canExecute: true,
            }
        },
    })
    
    console.log(`Ensuring boot log files exist:`)
    await FileSystem.ensureIsFile(pathToBootLogFile)
    console.log(`   - ${cyan(pathToBootLogFile)}`)
    await FileSystem.ensureIsFile(pathToOldBootLogFile)
    console.log(`   - ${cyan(pathToOldBootLogFile)}`)

// 
// 
// actually register the service
// 
// 
const onBootRegistration = `[Unit]
Description=CV Dark Boot
After=network.target

[Service]
ExecStart=sudo -- ${shellEscape(pathToActualInfinteRunScript)}

[Install]
WantedBy=default.target
`

const targetPath = `/etc/systemd/system/${nameOfStartService}.service`
console.log(`\nInstalling service\n    name: ${cyan(nameOfStartService)}\n    installing to: ${yellow(targetPath)}\n`)
// NOTE: I would normally do these things using JS/Deno, but they need sudo permission
//       and one way to do that is run them in a shell script
const actionsThatNeedSudo = `
    echo "pipefail means that if any command fails, the script will stop" > /dev/null
    set -o pipefail

    echo ${shellEscape(onBootRegistration)} > ${shellEscape(targetPath)}

    echo "#"
    echo "# starting services"
    echo "#"
    systemctl enable ${nameOfStartService}
    systemctl start ${nameOfStartService}
`

try {
    // NOTE: the run() command automatically shell-escapes its arguments
    var { code: exitCode } = await run`sudo bash -c ${actionsThatNeedSudo}`
} catch (error) {
    
}
var success = exitCode == 0
if (!success) {
    throw Error(`\n\nFailed: See above for details. For some reason the setup failed.`)
}

console.log(`

Done. Restart should now trigger the boot script to run in an infinite loop.
    ${green`boot script:`} ${pathToOnBootCommand}
    ${green`boot log:`} ${pathToBootLogFile}
    
`)
// (this comment is part of deno-guillotine, dont remove) #>
