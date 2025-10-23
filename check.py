import numpy as np
import time

# データ読み込み
full_array = np.load('full.npy')
lack_array = np.load('lack.npy')

# 1次元目のインデックス0を指定して実際のデータにアクセス
full = full_array[0]
lack = lack_array[0]

dist = 0
missing_indices = []

# 先頭から順に比較
i = 0
while i < len(lack):  # lackの長さまで比較
    if full[i+dist] != lack[i]:
        print(f"\nFound difference at index {i}")
        print(f"Full value at {i+dist}: {full[i+dist]}")
        print(f"Lack value at {i}: {lack[i]}")
        
        # 次にlack[i]と同じ値がfullのどこにあるか探す
        found = False
        for j in range(1, 200):  # 探索範囲を制限（例：次の10個まで）
            if i+dist+j < len(full) and lack[i] == full[i+dist+j]:
                print(f"Found matching value at distance {j}")
                missing_indices.append(i+dist)  # 欠落している要素のインデックスを記録
                dist += j
                found = True
                break
        
        if not found:
            print("No match found within next 100 elements")
            break
            
        time.sleep(1)
    i += 1

print("\nMissing indices:", missing_indices)
print("Total distance accumulated:", dist)
print("This should match the difference in lengths:", len(full) - len(lack))