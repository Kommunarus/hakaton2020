# hakaton2020

Компьютерное зрение реализовано на языке [Python 3.8](https://www.python.org/downloads/) с использованием двух фреймерков c открытой лицензией.
1. [YOLOv5](https://github.com/ultralytics/yolov5)
2. [Detectron2](https://github.com/facebookresearch/detectron2)

Необходимо склонировать оба проекта на свой компьютер и установить по инструкциям от каждой библиотеки.

Для детекции боксов изоляторов, гирлянд изоляторов, виброгасителей и столбов используется Yolo. 
Для детекции маски проводов используется Detectron2 (на основе ResNet).

Веса для YOLO и Detectron2 передаются отдельно.
Для YOLO папка веса располагается в 'yolov5/runs/exp17/' (https://yadi.sk/d/w-BAORcyEZrziw)
Для Detectron2 папка весов располагается в 'output/' (https://yadi.sk/d/lf5UjwxHywHgrg)

YOLO запускается через командную строку, например так:
    python detect.py --source ../img_test_izo --weights ./runs/exp17/weights/best.pt --conf 0.4 --save-txt --output ./out_box

Detectron2 так, пример:
    python detect_провода.py --output ./out_mask
    

После детекции, в директориях '--output xxxx', указанных при вызове скриптов, появятся текстовые файлы с боксами, масками, и обработанными фото с нарисованными боксами.

Зайдя на сайт [1С](https://v8.1c.ru/podderzhka-i-obuchenie/uchebnye-versii/distributiv-1s-predpriyatie-8-3-versiya-dlya-obucheniya-programmirovaniyu/) и скачайте учебную версию 1С.
Скачайте архив базы cpr.dt и разверните архив.
В конфигураторе обработки ... измените строчку... на... для запуска локального питона из 1с.

Их читает 1с...

 
