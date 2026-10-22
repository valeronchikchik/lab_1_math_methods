"""Лабораторна робота №1. Онлайн-калькулятор: метод Гаусса (СЛАР, визначник,
обернена матриця, число обумовленості). Варіант 15.

Запуск:  python app.py   →   http://127.0.0.1:5000
"""
import math
from flask import Flask, jsonify, request, Response

from gauss_core import (parse_matrix, parse_vector, gauss_solve, gauss_inverse,
                        cond_number, hilbert_system, mat_vec, mat_mul,
                        vec_norm_inf)

app = Flask(__name__)


def clean(o):
    """JSON не підтримує NaN/Infinity — замінюємо на None."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    return o


def fmt_matrix(M):
    return "\n".join(" ".join(repr(v) for v in row) for row in M)


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.post("/api/solve")
def api_solve():
    d = request.get_json(force=True)
    try:
        A = parse_matrix(d.get("matrix", ""))
        b = parse_vector(d.get("b", ""))
        n = len(A)
        if len(b) != n:
            raise ValueError(f"Вектор b має містити {n} елементів, а задано {len(b)}.")
        eps = float(d.get("eps", 1e-12))
        norm = d.get("norm", "rows")
        x_exact = parse_vector(d["x_exact"]) if d.get("x_exact", "").strip() else None
        if x_exact is not None and len(x_exact) != n:
            raise ValueError(f"Точний розв’язок має містити {n} елементів.")
    except Exception as e:                                   # помилка введення
        return jsonify({"error": str(e)}), 400

    res = gauss_solve(A, b, eps)
    out = {"n": n, "status": res["status"], "rank": res["rank"], "det": res["det"]}

    if res["status"] == "inconsistent":
        out["message"] = ("Система НЕСУМІСНА: ранг матриці A менший за ранг розширеної "
                          "матриці (A|b), розв’язків немає.")
    elif res["status"] == "infinite":
        out["message"] = (f"Система СУМІСНА, але невизначена: ранг = {res['rank']} < {n}, "
                          "розв’язків нескінченно багато (det A = 0).")
    else:
        x = res["x"]
        out["message"] = "Система сумісна і має єдиний розв’язок (det A ≠ 0)."
        out["x"] = x
        out["residual"] = vec_norm_inf([p - q for p, q in zip(mat_vec(A, x), b)])
        if x_exact is not None:
            dev = [abs(p - q) for p, q in zip(x, x_exact)]
            out["dev"] = dev
            out["dev_max"] = max(dev)
        c, na, ni, inv = cond_number(A, norm, eps)
        if inv is not None:
            out.update({"cond": c, "norm_A": na, "norm_inv": ni, "inverse": inv})
            E = mat_mul(A, inv)
            out["inv_check"] = max(abs(E[i][j] - (1.0 if i == j else 0.0))
                                   for i in range(n) for j in range(n))
    return jsonify(clean(out))


@app.get("/api/hilbert")
def api_hilbert():
    """Автоматичне задання A, b, x*: a_ij = 1/(i+j-1), x*_i = i^2, f = A x*."""
    m = int(request.args.get("m", 8))
    if not 2 <= m <= 40:
        return jsonify({"error": "m має бути в межах 2…40"}), 400
    A, f, xe = hilbert_system(m)
    return jsonify({"matrix": fmt_matrix(A), "b": " ".join(repr(v) for v in f),
                    "x_exact": " ".join(repr(v) for v in xe)})


@app.post("/api/study")
def api_study():
    """Дослідження: розв’язок системи з матрицею Гільберта для різних m."""
    d = request.get_json(force=True)
    norm = d.get("norm", "rows")
    try:
        ms = [int(v) for v in d.get("ms", [8, 10, 12, 14])]
        assert ms and all(2 <= m <= 40 for m in ms)
    except Exception:
        return jsonify({"error": "Некоректний список m (цілі числа 2…40)."}), 400
    rows = []
    for m in ms:
        A, f, xe = hilbert_system(m)
        # eps дуже малий: у матриці Гільберта ведучі елементи справді дуже малі,
        # але ненульові — їх не можна вважати нулями.
        res = gauss_solve(A, f, eps=1e-30)
        c, na, ni, _ = cond_number(A, norm, eps=1e-30)
        dev = [abs(p - q) for p, q in zip(res["x"], xe)]
        rows.append({"m": m, "cond": c, "det": res["det"], "x": res["x"],
                     "x_exact": xe, "dev": dev, "dev_max": max(dev),
                     "rel_dev": max(dev) / max(abs(v) for v in xe)})
    return jsonify(clean({"rows": rows}))


# ------------------------------------------------------------------ інтерфейс
INDEX_HTML = r"""<!doctype html>
<html lang="uk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Лаб. робота №1 — метод Гаусса</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1050px;margin:20px auto;padding:0 14px;color:#222}
 h1{font-size:1.4rem} h2{font-size:1.15rem;margin-top:0}
 section{border:1px solid #ccd;border-radius:8px;padding:14px 16px;margin:16px 0;background:#fafbff}
 textarea{width:100%;font-family:Consolas,monospace;font-size:13px;box-sizing:border-box}
 button{padding:6px 12px;margin:4px 6px 4px 0;cursor:pointer}
 label{font-size:.9rem} input[type=text],input[type=number],select{padding:4px}
 table{border-collapse:collapse;margin:8px 0;font-size:13px}
 td,th{border:1px solid #bbb;padding:3px 8px;text-align:right}
 th{background:#eef}
 .ok{color:#0a6b0a;font-weight:600} .bad{color:#b00020;font-weight:600} .warn{color:#a35b00;font-weight:600}
 .row{display:flex;gap:16px;flex-wrap:wrap} .row>div{flex:1;min-width:260px}
 .charts{display:flex;gap:16px;flex-wrap:wrap} .charts>div{flex:1;min-width:380px}
 small{color:#555}
</style></head><body>
<h1>Лабораторна робота №1. Розв’язання СЛАР методом Гаусса</h1>
<small>Онлайн-калькулятор: розв’язок СЛАР, визначник, обернена матриця, число обумовленості. Варіант 15.</small>

<section>
 <h2>1. Калькулятор</h2>
 <div class="row">
  <div><label>Матриця A (кожен рядок з нового рядка; числа через пробіл, допустимо 1/3)</label>
   <textarea id="A" rows="8"></textarea></div>
  <div><label>Вектор b (через пробіл)</label>
   <textarea id="b" rows="2"></textarea>
   <label>Точний розв’язок x* (необов’язково, для порівняння)</label>
   <textarea id="xe" rows="2"></textarea></div>
 </div>
 <div>
  <button onclick="loadV15()">Варіант 15 (тестова задача)</button>
  <label>m = <input type="number" id="mh" value="8" min="2" max="40" style="width:60px"></label>
  <button onclick="loadHilbert()">Автозаповнення (матриця Гільберта)</button>
  <label>Норма: <select id="norm"><option value="rows">max по рядках</option><option value="cols">max по стовпцях</option></select></label>
  <label>ε: <input type="text" id="eps" value="1e-12" style="width:70px"></label>
  <button onclick="solve()"><b>Розв’язати</b></button>
 </div>
 <div id="out"></div>
</section>

<section>
 <h2>2. Дослідження: матриця Гільберта, a<sub>ij</sub> = 1/(i+j−1), x*<sub>i</sub> = i², f = A·x*</h2>
 <label>Значення m (через кому): <input type="text" id="ms" value="8,10,12,14" style="width:160px"></label>
 <button onclick="study()"><b>Виконати дослідження</b></button>
 <div id="study"></div>
 <div class="charts"><div><canvas id="c1"></canvas></div><div><canvas id="c2"></canvas></div></div>
</section>

<script>
const V15={A:"2 -1 1 -1\n2 -1 0 -3\n3 0 -1 1\n2 2 -2 5",b:"0 -7 8 21",xe:"2 2 1 3"};
const $=id=>document.getElementById(id);
function fmt(v){ if(v===null||v===undefined) return '—'; if(v===0) return '0';
  const a=Math.abs(v); return (a<1e-4||a>=1e7)? v.toExponential(4) : String(+v.toFixed(8)); }
function loadV15(){ $('A').value=V15.A; $('b').value=V15.b; $('xe').value=V15.xe; $('out').innerHTML=''; }
async function loadHilbert(){
  const r=await fetch('/api/hilbert?m='+$('mh').value); const d=await r.json();
  if(d.error){alert(d.error);return;}
  $('A').value=d.matrix; $('b').value=d.b; $('xe').value=d.x_exact; $('out').innerHTML=''; }
function table(M){ return '<table>'+M.map(r=>'<tr>'+r.map(v=>'<td>'+fmt(v)+'</td>').join('')+'</tr>').join('')+'</table>'; }
async function solve(){
  const body={matrix:$('A').value,b:$('b').value,x_exact:$('xe').value,norm:$('norm').value,eps:$('eps').value};
  const r=await fetch('/api/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json(); let h='';
  if(d.error){ $('out').innerHTML='<p class="bad">Помилка введення: '+d.error+'</p>'; return; }
  const cls=d.status==='unique'?'ok':(d.status==='infinite'?'warn':'bad');
  h+='<p class="'+cls+'">'+d.message+'</p>';
  h+='<p>Ранг A = '+d.rank+'; det A = <b>'+fmt(d.det)+'</b></p>';
  if(d.x){
    h+='<p><b>Розв’язок x:</b></p>'+table([d.x]);
    h+='<p>Нев’язка ‖Ax−b‖<sub>∞</sub> = '+fmt(d.residual)+'</p>';
    if(d.dev) h+='<p>Відхилення |x − x*|:</p>'+table([d.dev])+'<p>max = '+fmt(d.dev_max)+'</p>';
    if(d.cond!==undefined){
      h+='<p>‖A‖ = '+fmt(d.norm_A)+'; ‖A⁻¹‖ = '+fmt(d.norm_inv)+'; число обумовленості cond(A) = <b>'+fmt(d.cond)+'</b></p>';
      h+='<p><b>Обернена матриця A⁻¹:</b></p>'+table(d.inverse)+'<p>Перевірка ‖A·A⁻¹ − E‖<sub>∞</sub> = '+fmt(d.inv_check)+'</p>';
    }
  }
  $('out').innerHTML=h;
}
let ch1,ch2;
async function study(){
  const ms=$('ms').value.split(/[,;\s]+/).filter(Boolean).map(Number);
  const r=await fetch('/api/study',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ms,norm:$('norm').value})});
  const d=await r.json(); if(d.error){ $('study').innerHTML='<p class="bad">'+d.error+'</p>'; return; }
  let h='<table><tr><th>m</th><th>det A</th><th>cond(A)</th><th>max |x − x*|</th><th>відносне відхилення</th></tr>';
  d.rows.forEach(o=>{ h+='<tr><td>'+o.m+'</td><td>'+fmt(o.det)+'</td><td>'+fmt(o.cond)+'</td><td>'+fmt(o.dev_max)+'</td><td>'+fmt(o.rel_dev)+'</td></tr>'; });
  $('study').innerHTML=h+'</table>';
  const colors=['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd','#8c564b','#17becf','#7f7f7f'];
  if(ch1)ch1.destroy(); if(ch2)ch2.destroy();
  const maxm=Math.max(...d.rows.map(o=>o.m));
  ch1=new Chart($('c1'),{type:'line',data:{labels:Array.from({length:maxm},(_,i)=>i+1),
    datasets:d.rows.map((o,k)=>({label:'m = '+o.m,data:o.dev.map(v=>v>0?v:null),borderColor:colors[k%8],backgroundColor:colors[k%8],tension:.1}))},
    options:{plugins:{title:{display:true,text:'Відхилення |x_i − x*_i| по компонентах'}},
      scales:{y:{type:'logarithmic',title:{display:true,text:'абсолютне відхилення'}},x:{title:{display:true,text:'i'}}}}});
  ch2=new Chart($('c2'),{type:'line',data:{labels:d.rows.map(o=>o.m),datasets:[
    {label:'max |x − x*|',data:d.rows.map(o=>o.dev_max),borderColor:'#d62728',backgroundColor:'#d62728'},
    {label:'cond(A)',data:d.rows.map(o=>o.cond),borderColor:'#1f77b4',backgroundColor:'#1f77b4'}]},
    options:{plugins:{title:{display:true,text:'Відхилення та число обумовленості залежно від m'}},
      scales:{y:{type:'logarithmic'},x:{title:{display:true,text:'m'}}}}});
}
loadV15();
</script></body></html>
"""

if __name__ == "__main__":
    app.run(debug=True)