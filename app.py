from flask import Flask, render_template, request, jsonify
from library import PdfNavigator
from dotenv import load_dotenv
import os, warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
navigator = None

# .env 파일에서 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# 업로드된 파일을 저장할 폴더 설정 (기본값 'uploads')
upload_folder = os.getenv("UPLOAD_FOLDER", "uploads")
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    global navigator
    if 'file' not in request.files:
        return "파일이 없습니다.", 400
    
    file = request.files['file']
    if file.filename == '':
        return "선택된 파일이 없습니다.", 400
    
    if file and file.filename.lower().endswith('.pdf'):
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        
        try:
            # PdfNavigator를 사용하여 PDF 분석 (번역 및 요약)
            navigator = PdfNavigator(file_path)
            result = navigator.run()
            
            # 파일 분석 후 삭제 (공간 절약)
            # os.remove(file_path)
            
            # 분석 결과를 결과 페이지에 전달하여 렌더링
            return render_template("result.html", 
                                 summary=result.get('summary', '요약 결과가 없습니다.'), 
                                 translation=result.get('translation', '번역 결과가 없습니다.'),
                                 file_path=file_path)
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return f"분석 중 오류 발생: {str(e)}", 500
    
    return "PDF 파일만 업로드 가능합니다.", 400

@app.route("/ask", methods=["POST"])
def ask():
    # PDF 내용을 바탕으로 질의응답 처리
    data = request.get_json()
    file_path = data.get("file_path")
    question = data.get("question")
        
    try:
        # PdfNavigator를 사용하여 질의응답 (RAG)
        result = navigator.query(question)
        return jsonify({"answer": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)