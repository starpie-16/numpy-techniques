import numpy as np

def co_occurrence_matrix(indexed_data, vocab_size):
    """
    indexed_data: Mảng 1D chứa toàn bộ các từ đã được số hóa trong văn bản.
                  Ví dụ: np.array([0, 1, 2, 1, 0, 2])
    vocab_size: Kích thước từ điển V.
    
    Nhiệm vụ: Trả về ma trận 2D shape (V, V) đếm tần suất đồng xuất hiện 
              của các từ cạnh nhau (cửa sổ ngữ cảnh = 1).
    """
   indexed_data = np.array(indexed_data)
   V = vocab_size

   words_left = indexed_data[:-1]
   words_right = indexed_data[1:]

   flat_indices = words_left * V + words_right

   cooc_flat = np.bincount(flat_indices, minlength = V*V)

   final_matrix = cooc_flat.reshape(V, V)

   return final_matrix
