# hakaton2020
## Установка python
Компьютерное зрение реализовано на языке [Python 3.8](https://www.python.org/downloads/) с использованием двух фреймерков c открытой лицензией.
1. [YOLOv5](https://github.com/ultralytics/yolov5)
2. [Detectron2](https://github.com/facebookresearch/detectron2)

Необходимо склонировать оба проекта на свой компьютер и установить по инструкциям от каждой библиотеки.
Скопируйте файл из корня этого проекта detect_провода.py для запуска детекции масок проводов.
Для детекции боксов изоляторов, гирлянд изоляторов, виброгасителей и столбов используется Yolo. 
Для детекции маски проводов используется Detectron2 (на основе ResNet).
В настроечных файлах йоло (yolov5/models/yolov5s.yaml) поставить число классов nc=4. Во время обучения сети использовался файл rosseti50_izo.yaml. Этот настроечный файл находился в пути (yolov5/data/rosseti50_izo.yaml)

Веса для YOLO и Detectron2 передаются отдельно.
Для YOLO веса поместите в 'yolov5/runs/exp17/weights' [скачать](https://yadi.sk/d/w-BAORcyEZrziw).
Для Detectron2 файл весов поместите в 'output/' [скачать](https://yadi.sk/d/lf5UjwxHywHgrg)

Для проверки работы скриптов:
YOLO запускается через командную строку, например так:
    python detect.py --source ../img_test_izo --weights ./runs/exp17/weights/best.pt --conf 0.4 --save-txt --output ./out_box

Detectron2 так, пример:
    python detect_провода.py --output ./out_mask
    

После детекции, в директориях '--output xxxx', указанных при вызове скриптов, появятся текстовые файлы с боксами, масками, и обработанными фото с нарисованными боксами.

## Установка 1С

Зайдя на сайт [1С](https://v8.1c.ru/podderzhka-i-obuchenie/uchebnye-versii/distributiv-1s-predpriyatie-8-3-versiya-dlya-obucheniya-programmirovaniyu/) и скачайте учебную версию 1С.
Скачайте архив базы [cpr.dt](https://cloud.mail.ru/public/3rB9/3ucz3up21) и разверните архив.
В конфигураторе обработки ... измените строчку... на... для запуска локального питона из 1с.
Запустите 1С в режиме предприятия...


 
