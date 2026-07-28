#!/usr/bin/env python3
"""
将易理探讨15-八卦相荡_v3_ALFWorld.md 转换为 docx 格式
"""
import sys, os, re
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

SOURCE = os.path.expanduser('~/MXL/科研/ylyw/QYUF/paper/易理探讨15-八卦相荡_v3_ALFWorld.md')
OUTPUT = os.path.expanduser('~/MXL/科研/ylyw/QYUF/paper/易理探讨15-八卦相荡_v3_ALFWorld.docx')

def parse_md(md_text):
    """简单的md解析，保留标题、段落、表格、代码块"""
    lines = md_text.split('\n')
    blocks = []
    in_table = False
    table_lines = []
    in_code = False
    code_lines = []
    
    for line in lines:
        # 代码块
        if line.startswith('```'):
            if in_code:
                blocks.append({'type': 'code', 'text': '\n'.join(code_lines)})
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        
        # 表格
        if line.startswith('|') and '|' in line[1:]:
            if not in_table:
                table_lines = [line]
                in_table = True
            else:
                table_lines.append(line)
            continue
        elif in_table:
            if line.strip() and line.startswith('|'):
                table_lines.append(line)
                continue
            else:
                blocks.append({'type': 'table', 'lines': table_lines})
                table_lines = []
                in_table = False
                if line.strip():
                    pass  # fall through
        
        if in_table:
            continue
        
        # 空行
        if not line.strip():
            continue
        
        # 标题
        if line.startswith('#'):
            level = len(re.match(r'^#+', line).group())
            title = line.lstrip('#').strip()
            blocks.append({'type': 'heading', 'level': min(level, 6), 'text': title})
            continue
        
        # 水平线
        if line.strip().startswith('---') or line.strip().startswith('==='):
            blocks.append({'type': 'hr'})
            continue
        
        # 列表
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            blocks.append({'type': 'list_item', 'text': line.strip()[2:], 'level': 0})
            continue
        
        # 数字列表
        if re.match(r'^\d+[.)]', line.strip()):
            blocks.append({'type': 'list_item', 'text': re.sub(r'^\d+[.)]\s*', '', line.strip()), 'level': 0})
            continue
        
        # 普通段落
        blocks.append({'type': 'paragraph', 'text': line})
    
    return blocks


def format_paragraph_text(paragraph, text):
    """支持粗体/斜体/等宽文本"""
    # 替换 **bold** → 加粗
    parts = re.split(r'(\*\*.*?\*\*|`[^`]+`|\$[^$]+\$)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith('`') and part.endswith('`'):
            run = paragraph.add_run(part[1:-1])
            run.font.name = 'Courier New'
            run.font.size = Pt(9)
        elif part.startswith('$') and part.endswith('$'):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        else:
            paragraph.add_run(part)


def generate_docx():
    with open(SOURCE, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    blocks = parse_md(md_text)
    
    doc = Document()
    
    # 设置默认字体
    style = doc.styles['Normal']
    font = style.font
    font.name = '宋体'
    font.size = Pt(11)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    
    # 设置页边距
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(3.18)
        section.right_margin = Cm(3.18)
    
    heading_styles = ['Heading 1', 'Heading 2', 'Heading 3', 'Heading 4', 'Heading 5', 'Heading 6']
    
    i = 0
    while i < len(blocks):
        b = blocks[i]
        
        if b['type'] == 'heading':
            level = b['level']
            style_name = heading_styles[min(level, 6) - 1]
            p = doc.add_heading(b['text'], level=min(level, 6))
        
        elif b['type'] == 'paragraph':
            text = b['text']
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.5
            
            # 处理特殊前缀
            if text.startswith('**表') and '**' in text[3:]:
                # 表格标题
                run = p.add_run(text)
                run.bold = True
            elif '：' in text and len(text) < 60:
                # 可能是关键定义
                format_paragraph_text(p, text)
            else:
                format_paragraph_text(p, text)
        
        elif b['type'] == 'code':
            p = doc.add_paragraph()
            run = p.add_run(b['text'])
            run.font.name = 'Courier New'
            run.font.size = Pt(8)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
        
        elif b['type'] == 'hr':
            p = doc.add_paragraph()
            run = p.add_run('─' * 50)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        elif b['type'] == 'list_item':
            p = doc.add_paragraph(style='List Bullet')
            format_paragraph_text(p, b['text'])
        
        elif b['type'] == 'table':
            lines = b['lines']
            if len(lines) <= 2:
                i += 1
                continue
            
            # 跳过分隔行（|----|---）
            data_rows = [l for l in lines if l.strip() and not all(c in ' |-:' for c in l)]
            
            rows = []
            for line in data_rows:
                cells = [c.strip() for c in line.split('|')[1:-1]]
                rows.append(cells)
            
            if rows:
                table = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
                table.style = 'Light Grid Accent 1'
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                
                for ri, row_data in enumerate(rows):
                    for ci, cell_text in enumerate(row_data):
                        if ci < len(table.columns):
                            cell = table.cell(ri, ci)
                            cell.text = cell_text
                            # 表头加粗
                            if ri == 0:
                                for paragraph in cell.paragraphs:
                                    for run in paragraph.runs:
                                        run.bold = True
            
            doc.add_paragraph()  # 表后间距
        
        i += 1
    
    # 保存
    doc.save(OUTPUT)
    print(f"已保存: {OUTPUT}")


if __name__ == '__main__':
    generate_docx()
