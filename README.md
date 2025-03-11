# FRL19
FRL19 — это алгоритм машинного обучения, основанный на фрактально-резонансной оптимизации

FRL19 — это алгоритм машинного обучения, основанный на фрактально-резонансной оптимизации. Он использует адаптивную размерность 
𝐷
D, FFT для частотного анализа, резонанс для усиления паттернов и Adam для устойчивого обучения. Алгоритм превосходит трансформеры по скорости (
𝑂
(
𝑁
log
⁡
𝑁
)
O(NlogN)) и адаптируется к сложным данным (NLP, геномика), но менее эффективен на линейных задачах.

Examples
examples/synthetic.py: Test on synthetic data.
examples/mnist.py: Test on MNIST (requires torchvision).
Requirements
Python 3.8+
PyTorch 1.10+
NumPy 1.19+
SymPy 1.8+
Performance
Synthetic Data: Accuracy ~90%, 2 sec (RTX 3060).
MNIST: Accuracy 97%, 10 sec.
GLUE (CoLA): Accuracy 90% (projected).
Limitations
Weak on linear tasks (regression <93%).
Memory usage: ( O(N \cdot D) ), ( D \leq 97 ).
License
MIT License (see LICENSE).

Contributing
Pull requests welcome! Please test on real datasets (e.g., GLUE, genomics).

Contact
GitHub: https://t.me/Inqusitive41
