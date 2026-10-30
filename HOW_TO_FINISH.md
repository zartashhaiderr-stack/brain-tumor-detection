# How to finish and deploy (step by step)

## 1. Train the model in Google Colab
1. Open https://colab.research.google.com, then File > Upload notebook, and pick `notebooks/brain_tumor_detection_colab.ipynb`.
2. Runtime > Change runtime type > **T4 GPU** > Save.
3. Optional first check: in Step 1 set `QUICK_TEST = True`, Runtime > Run all (about 5 minutes). If it finishes without errors, set it back to `False`.
4. Runtime > Run all. Wait about 30 to 40 minutes. `project_outputs.zip` downloads at the end.
5. Unzip it **into this repo folder**. It fills `weights/`, `results/`, `web/model/`, `web/samples/`, `web/results/` and creates `web/project.json`.

## 2. Put it on GitHub
1. Create a new **public** repository on github.com (for example `brain-tumor-detection`).
2. Click "uploading an existing file" and drag in everything from this folder (all files are under GitHub's 25 MB browser upload limit).
3. Commit.

## 3. Deploy on Vercel
1. Go to https://vercel.com, sign in with GitHub, click **Add New > Project**, and import the repository.
2. Set **Root Directory** to `web`. Framework preset: **Other**. No build command.
3. Click Deploy. Open the link and test a sample scan.

## 4. Add your links
1. In `web/index.html`, find the comment `EDIT YOUR LINKS HERE` and replace each `href="#"` with your real links (GitHub, report, slides, YouTube, LinkedIn).
2. In `README.md`, replace the `ADD_..._LINK` lines.
3. Commit. Vercel updates the site automatically.

## 5. Check before submitting
- The site loads and a sample scan gives a result.
- `https://YOUR-SITE.vercel.app/project.json` opens and shows your scores.
- On GitHub, `weights/brain_tumor_effnetb0.onnx` and `.keras` show their real size (about 16 to 17 MB), not a few KB.
