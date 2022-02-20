# Camera-tracking-app

Приложение для трэкинга камеры: получаем на вход последовательность кадров/видео, на выходе получаем координаты трэка камеры, которая осуществляла съёмку, и координаты объектов сцены.

# Установка зависимостей

```bash
pip install -r requirements.txt
```

# Поиск уголков на одном примере:
```bash
python corners.py --show <relative path to video file> --dump-corners <relative path to dump corners in .pickle file, optional> --load-corners <relative path to get corners from .pickle file, optional> --help <output helping message then quit, optional>
```

# Трекинг уголков:
```bash
python camtrack.py --show --load-corners <relative path to get corners from .pickle file> --camera-poses <relative path to get init camera coordinates from .yml file, optional> --frame-1 <the first frame to compute camera from, non-negative number, optional> —frame-2 <the second frame to compute camera from, non-negative number, optional> <relative path to input video file> dataset/fox_head_short/camera.yml tracks.yml points.yml
```

# Визуализация трека камеры в 3D:
```bash
python render.py dataset/fox_head_short/camera.yml tracks.yml points.yml
```
