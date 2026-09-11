"""
================================================================================
 生产工程补齐 · 补2 · 文档解析：PDF 文本 + 表格 + 扫描件 OCR（RAG 的真正入口）
================================================================================
 补上「RAG 全流程里最脏最难的一段——文档解析」。之前的 RAG 案例输入都是现成文本，
 真实企业知识库是一堆 PDF/Word/扫描件，得先把它们变成干净可切块的文本：
   ① 文本 PDF：PyMuPDF(fitz) 逐页 get_text()——最快最稳的纯文本抽取。
   ② 表格：page.find_tables() 把表格结构抽成行列(合同/报表全靠它)，转成 Markdown 喂 LLM。
   ③ 扫描件(图片 PDF)：没有文字层，必须先渲染成图再 OCR(pytesseract)——本机未装则给降级说明。
   ④ 解析后 → 清洗 → 切块，才接上前面的 RAG(嵌入→检索→生成)。
 为什么难：PDF 是「打印格式」不是「结构化数据」，跨栏/表格/页眉页脚/扫描件都是坑；
   生产选型：文本层用 PyMuPDF/pdfplumber，复杂版式/多格式用 Unstructured，扫描件叠 OCR。
 依赖：fitz(PyMuPDF，本机已装真跑)；pytesseract/unstructured 未装(降级为说明)。
 跑：python3 补2_文档解析_PDF表格OCR.py   （本文件会先用 fitz 造一个样例 PDF 再解析，自包含）
     （注：按用户要求本文件未在本机执行，仅作真实可跑代码）
================================================================================
"""
import fitz  # PyMuPDF


SAMPLE_PDF = "/tmp/_sample_contract.pdf"


# ==============================================================================
# ① 造一个样例 PDF（自包含，免依赖外部文件）：一段正文 + 一个表格
# ==============================================================================
def make_sample_pdf(path=SAMPLE_PDF):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "合同编号：HT-2024-001\n甲方：某某科技有限公司\n乙方：另一家公司\n"
                               "标的：AI 推理服务采购", fontsize=12)
    # 画一个简单表格(用线+文字模拟，真实 PDF 表格也是这种「线框+文本」)
    rows = [("项目", "数量", "单价"), ("GPU 卡", "8", "￥50000"), ("服务费", "1", "￥120000")]
    x0, y0, dx, dy = 72, 160, 120, 24
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((x0 + c * dx, y0 + r * dy), cell, fontsize=11)
    doc.save(path)
    doc.close()
    return path


# ==============================================================================
# ② 解析：逐页抽文本 + 抽表格
# ==============================================================================
def parse_pdf(path):
    doc = fitz.open(path)
    result = {"text": [], "tables": []}
    for page in doc:
        result["text"].append(page.get_text())          # 纯文本层
        try:
            tf = page.find_tables()                       # PyMuPDF>=1.23 内置表格识别
            for t in tf.tables:
                result["tables"].append(t.extract())      # 行列二维数组
        except Exception:
            pass                                          # 老版本无 find_tables，跳过
    doc.close()
    return result


def table_to_markdown(rows):
    """把抽出的行列数组转成 Markdown 表格（喂 LLM 最友好的表格格式）。"""
    if not rows:
        return ""
    head = "| " + " | ".join(str(c or "") for c in rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(str(c or "") for c in r) + " |" for r in rows[1:]]
    return "\n".join([head, sep, *body])


# ==============================================================================
# ③ 扫描件 OCR：无文字层的图片 PDF，渲染成图再 OCR（本机未装 pytesseract → 降级说明）
# ==============================================================================
def ocr_pdf(path):
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        return ("（未装 pytesseract/Pillow；生产扫描件流程：page.get_pixmap(dpi=300) 渲染成图 → "
                "pytesseract.image_to_string(img, lang='chi_sim+eng') → 得到文字层，再走后续切块。"
                "系统需先装 tesseract-ocr 引擎 + 中文语言包。）")
    doc = fitz.open(path)
    out = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)                    # 渲染成高清位图
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        out.append(pytesseract.image_to_string(img, lang="chi_sim+eng"))
    doc.close()
    return "\n".join(out)


# ==============================================================================
# ④ 解析结果 → 清洗 → 切块（接上前面的 RAG：切块后嵌入入库）
# ==============================================================================
def to_chunks(text, size=200, overlap=40):
    text = " ".join(text.split())                          # 压掉多余空白/换行
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap                                # 重叠切块，防切断语义
    return chunks


def main():
    path = make_sample_pdf()
    parsed = parse_pdf(path)
    full_text = "\n".join(parsed["text"])
    md_tables = [table_to_markdown(t) for t in parsed["tables"]]
    chunks = to_chunks(full_text)
    ocr_note = ocr_pdf(path)  # 本机无 OCR 引擎 → 返回降级说明字符串
    # full_text 含「合同编号 HT-2024-001…」；md_tables 为 Markdown 表格；chunks 为重叠切块
    assert "HT-2024-001" in full_text and len(chunks) >= 1
    print(f"✅ 补2 跑通：PDF 抽文本 {len(full_text)} 字 / 表格 {len(parsed['tables'])} 个 / 切块 {len(chunks)} 段"
          f"（OCR：{'已装真跑' if '未装' not in ocr_note else '未装→降级说明'}）")
    # 面试：Q PDF 解析难在哪? A 是打印格式非结构化，跨栏/表格/扫描件都是坑;
    #      Q 扫描件怎么处理? A 无文字层，渲染成图 + OCR; Q 表格怎么喂 LLM? A 抽成行列转 Markdown。


if __name__ == "__main__":
    main()
