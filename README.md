# israr-voice-classifier
Voice Command Recognition Model for ESP32 – Part of the Israr Assistive Technology Projec

**Israr** is an assistive technology project designed to help people with visual impairments interact with digital platforms through **voice commands**.  
This repository contains the machine learning model I trained and deployed on the **ESP32 microcontroller** for real-time voice command recognition.

### 🧠 Model Overview
The model was developed using **TensorFlow** and **Keras**, built as a **1D Convolutional Neural Network (CNN)** optimized for embedded systems.  
To improve generalization, I used techniques like **SpecAugment**, **Batch Normalization**, and **Dropout**.  
The model was then **quantized (int8)** for efficient deployment on low-power hardware such as ESP32.

### 🎯 Commands Recognized
The model can recognize six spoken commands:
- `password`
- `read`
- `stopread`
- `username`
- `zoom_in`
- `zoom_out`

### 📊 Model Performance
| Metric | Validation | Test |
|--------|-------------|------|
| Accuracy | 91.3% | **97.0%** |
| F1-score | 0.91 | **0.97** |
| ROC-AUC | 0.993 | **0.999** |

These results confirm the model’s robustness and readiness for real-world applications in **assistive voice control**.

### ⚙️ Tech Stack
- TensorFlow / Keras  
- Edge Impulse  
- ESP32  
- TensorFlow Lite (quantization)  
- Python

---

## 🇸🇦 الوصف بالعربية

**إصرار (Israr)** هو مشروع تكنولوجي موجه لذوي الإعاقة البصرية، يهدف إلى تمكينهم من التفاعل مع المنصات الرقمية باستخدام **الأوامر الصوتية**.  
في هذا المشروع، قمتُ بتدريب نموذج ذكاء اصطناعي للتعرف على الأوامر الصوتية وتشغيله على **قطعة ESP32**.

### 🧠 تفاصيل النموذج
تم بناء النموذج باستخدام مكتبات **TensorFlow** و **Keras** كشبكة عصبية تلافيفية (CNN)،  
واستخدمت تقنيات مثل **SpecAugment** لتحسين جودة البيانات و **Dropout** و **Batch Normalization** لزيادة دقة النموذج.  
بعد التدريب، تم **ضغط النموذج وتحويله إلى صيغة int8** ليعمل بكفاءة على الأجهزة منخفضة الطاقة مثل ESP32.

### 🎯 الأوامر التي يتعرف عليها النموذج
- كلمة المرور `password`  
- القراءة `read`  
- إيقاف القراءة `stopread`  
- اسم المستخدم `username`  
- تكبير الشاشة `zoom_in`  
- تصغير الشاشة `zoom_out`

### 📈 النتائج
حقق النموذج دقة اختبار وصلت إلى **97%** ودرجة F1 بلغت **0.97**،  
مما يثبت فعاليته في التطبيقات المساعدة لضعاف البصر.

---

### 📂 Repository Contents
