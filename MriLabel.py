import os
from collections import Counter

label_dir = 'C:/Users/sepehr/Desktop/projects/MRI/Bone Fractures Detection/train/labels'  # مسیر پوشه label های validation
class_counter = Counter()

for fname in os.listdir(label_dir):
    if fname.endswith('.txt'):
        with open(os.path.join(label_dir, fname)) as f:
            for line in f:
                cls = int(line.split()[0])
                class_counter[cls] += 1

print("نمونه‌های هر کلاس در validation:")
for i in range(10):
    print(f"{i}: {class_counter[i]}")
