Setting Up rclone to Sync Brain Sections and Annotations
========================================================

We use a tool called **rclone** to move files between your computer and our
shared Google Drive. It only transfers files that are *new or changed*, so it's
much faster than re-downloading everything, and it's how you'll get the section
images and upload your annotations.

This guide walks you through the one-time setup. It looks long only because we
explain each prompt — the actual process takes about five minutes. Follow it
carefully and it will just work.

> **Tip:** rclone runs in a **terminal** (Command Prompt / PowerShell on
> Windows, Terminal on Mac). You type commands and answer questions. When a
> prompt shows options like `y/n>`, you type one letter and press Enter.


---

Part 1 — Install rclone
-----------------------

**Windows**
1. Go to <https://rclone.org/downloads/> and download the "Windows AMD64" zip.
2. Unzip it somewhere you'll remember (e.g. your Desktop).
3. Open that unzipped folder, then open a terminal *in that folder*: click the
   address bar at the top of the file window, type `cmd`, and press Enter. A
   terminal opens already pointed at the folder.
4. Test it by typing:

       rclone version

   If you see version information, it works.

**Mac**
1. Open Terminal (find it with Spotlight: press Cmd+Space, type "Terminal").
2. Install with one command (this uses Homebrew; if you don't have Homebrew,
   follow the "Mac" instructions at <https://rclone.org/install/>):

       brew install rclone

3. Test it:

       rclone version

If `rclone version` prints version info, you're ready.


---

Part 2 — Connect rclone to Google Drive
---------------------------------------

This is the one-time setup that links rclone to your Google account. Start it by
typing:

    rclone config

You'll now answer a series of questions. Here's exactly what to do at each one.

### 1. Make a new remote

You'll see options like `n) New remote`, `q) Quit`. Type:

    n

("remote" is just rclone's word for a saved connection.)

### 2. Name it

It asks for a name. Type exactly:

    gdrive

> Use this exact name — the commands later in this guide assume it. Don't use
> spaces or capital letters.

### 3. Choose the storage type

A long numbered list of storage providers appears. **Google Drive is number 18.**
Type:

    18

> If for some reason the number is different on your screen, you can instead
> type the word `drive` and press Enter — that always selects Google Drive.

### 4. Client ID and Client Secret — leave both blank

It asks for `client_id` and then `client_secret`. **Leave both empty** — just
press Enter at each one. (These are optional advanced settings we don't need.)

### 5. Choose the scope

It asks what level of access rclone should have, with a numbered list. Type:

    1

> **Why 1?** Option 1 is "Full access," which lets you both **download**
> sections and **upload** your annotations. (There's a read-only option, but you
> need to upload, so you want full access.) This only affects *your own* Google
> account — it doesn't give anyone access to your files.

### 6. root_folder_id and service_account_file — leave blank

It may ask for a `root_folder_id` and a `service_account_file`. **Leave both
blank** — press Enter at each.

### 7. Advanced config? — No

When it asks `Edit advanced config?`, type:

    n

### 8. Use auto config? — Yes

When it asks `Use auto config?`, type:

    y

Your web browser will open automatically.

### 9. Log in and allow access (in the browser)

- Log in with **the Google account we shared the brain-sections folder with**
  (use the email address you gave us).
- You'll see a screen saying "rclone wants access to your Google Account."
  This is expected — click **Allow / Continue**.
- If you see a warning that the app "isn't verified," that's normal for rclone.
  Click the small "Advanced" or "Continue" link to proceed, then Allow.
- When it says you can close the browser tab, do so and return to the terminal.

> This step is what actually connects rclone to *your* Drive. The permission
> lives only on your computer, and you can remove it anytime from your Google
> Account's security settings.

### 10. Configure this as a Shared Drive (Team Drive)? — No

When it asks `Configure this as a Shared Drive (Team Drive)?`, type:

    n

> Our folder is a normal shared *folder*, not a Google "Shared Drive" (a special
> team-owned space). So the answer is **No**. (These sound the same but aren't —
> just answer No.)

### 11. Confirm and quit

- It shows a summary and asks `Keep this remote?` — type `y`.
- Back at the main menu, type `q` to quit.

Setup is done. You only do Part 2 once.


---

Part 3 — Check that it works
----------------------------

List the folders in the Drive to confirm the connection:

    rclone lsd gdrive:

You should see the shared folders listed (for example, a `sections` folder and
an `annotations` folder). If you see them, everything is connected.

> `gdrive:` means "the Drive I just connected." The part after the colon is a
> folder path inside that Drive — so `gdrive:sections` is the sections folder.
> You do **not** use the long Google Drive web address anywhere; just the folder
> name.


---

Part 4 — Downloading the sections
---------------------------------

To download the section images into a folder on your computer (this creates a
local folder called `sections` if it doesn't exist):

    rclone copy gdrive:sections ./sections --progress

- `gdrive:sections` is the source (on Drive).
- `./sections` is the destination (a folder on your computer).
- `--progress` shows a live progress bar.

Run this again anytime to grab any **new** sections — rclone skips everything you
already have and only downloads what's changed.


---

Part 5 — Uploading your annotations
-----------------------------------

When you've made annotations, upload them like this (replace `yourname` with the
name we gave you):

    rclone copy ./my-annotations gdrive:annotations/yourname --progress

- `./my-annotations` is your local folder of annotation files (the source).
- `gdrive:annotations/yourname` is your own folder on Drive (the destination).

> **Please use `copy`, not `sync`, for uploading.** `copy` adds your new and
> changed files without deleting anything. (`sync` would try to make the Drive
> folder *identical* to your local folder, which could delete other files —
> avoid it unless told otherwise.)

> **Always upload into your own `annotations/yourname` folder**, so everyone's
> work stays separate and no one overwrites anyone else.


---

Part 6 — The safety habit: `--dry-run`
--------------------------------------

If you're ever unsure what a command will do, add `--dry-run` to the end. It
shows you exactly what *would* happen — what would upload or download — **without
actually changing anything**:

    rclone copy ./my-annotations gdrive:annotations/yourname --dry-run

Read the output, make sure it looks right, then run the command again *without*
`--dry-run` to do it for real. When in doubt, dry-run first.


---

Quick reference
---------------

    rclone lsd gdrive:                                   # list Drive folders
    rclone copy gdrive:sections ./sections --progress   # download sections
    rclone copy ./my-annotations gdrive:annotations/yourname --progress   # upload
    (add --dry-run to any command to preview it safely)

If a command gives an error, read the message — rclone's errors usually say what
went wrong. If you're stuck, copy the full command you ran and the full error
text and send it to us, and we'll help.
