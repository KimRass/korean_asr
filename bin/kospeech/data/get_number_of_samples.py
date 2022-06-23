from pathlib import Path

data_dir = Path("/Users/jongbeom.kim/Documents/ksponspeech")

ls_pcm = list(data_dir.glob("*/*/*/*/*.pcm"))
len(ls_pcm)