import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="GharIQ - India House Price API")
model = joblib.load("models/model.joblib")


class HouseFeatures(BaseModel):
    city: str
    locality: str = "Unknown"
    bhk: float
    bathrooms: float
    area_sqft: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(f: HouseFeatures):
    row = {
        "city": f.city,
        "locality": f.locality,
        "bhk": f.bhk,
        "bathrooms": f.bathrooms,
        "area_sqft": f.area_sqft,
    }
    price = float(model.predict(pd.DataFrame([row]))[0])
    return {"predicted_price_inr": round(price, 2)}


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GharIQ &mdash; ghar ka smart estimate</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,ital,wght@9..144,0,500;9..144,0,600;9..144,0,700;9..144,1,600&display=swap" rel="stylesheet">
<style>
  :root{
    --canvas:#0b1917;--panel:#122220;--panel-hi:#16302b;--line:#234440;--line-soft:#1b332f;
    --ink:#eef4f2;--muted:#88a59f;--gold:#e8b04b;--gold-soft:rgba(232,176,75,.10);--sea:#5fb6a8;
    --serif:"Fraunces",Georgia,"Times New Roman",serif;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;background:
      radial-gradient(1100px 600px at 88% -12%,#1d3b37 0%,transparent 58%),
      radial-gradient(700px 500px at -5% 110%,#16302c 0%,transparent 55%),var(--canvas);
    color:var(--ink);font-family:var(--sans);display:flex;justify-content:center;padding:34px 22px 60px}
  .wrap{width:100%;max-width:820px}

  .brand{display:flex;align-items:baseline;gap:10px;margin:0 0 20px}
  .mark{width:30px;height:30px;flex:0 0 auto;align-self:center}
  .word{font-family:var(--serif);font-weight:600;font-size:22px;color:var(--ink);letter-spacing:-.01em}
  .word b{color:var(--gold);font-weight:600}
  .tag{font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--sea)}
  h1{font-family:var(--serif);font-optical-sizing:auto;font-weight:600;font-size:38px;line-height:1.03;
    margin:0 0 8px;letter-spacing:-.015em}
  h1 em{font-style:italic;color:var(--gold)}
  .sub{color:var(--muted);margin:0 0 24px;font-size:15px;max-width:56ch}

  .card{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:22px 24px 26px}

  .modebar{display:flex;gap:8px;background:var(--canvas);border:1px solid var(--line);border-radius:12px;padding:4px;margin-bottom:22px}
  .modebtn{flex:1;border:none;background:transparent;color:var(--muted);font-family:var(--sans);font-size:14px;font-weight:600;
    padding:10px;border-radius:9px;cursor:pointer;transition:background .12s,color .12s}
  .modebtn[aria-selected="true"]{background:var(--gold-soft);color:var(--gold)}

  .homes{display:grid;gap:22px}
  .homes.compare{grid-template-columns:1fr 1fr}
  .home{display:grid;gap:16px}
  .home>.htitle{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--sea);font-weight:700}
  .field>label{display:block;font-size:12.5px;letter-spacing:.04em;color:var(--muted);margin-bottom:8px}
  .field>label .opt{color:var(--line);font-size:11px;letter-spacing:.06em;text-transform:uppercase;margin-left:6px}
  input[type=number],input[type=text],select{width:100%;background:var(--canvas);border:1px solid var(--line);
    color:var(--ink);border-radius:11px;padding:11px 13px;font-size:15px;font-variant-numeric:tabular-nums;font-family:var(--sans)}
  select{cursor:pointer;appearance:none;background-image:linear-gradient(45deg,transparent 50%,var(--sea) 50%),linear-gradient(135deg,var(--sea) 50%,transparent 50%);
    background-position:calc(100% - 18px) 18px,calc(100% - 13px) 18px;background-size:5px 5px;background-repeat:no-repeat;padding-right:36px}
  input:focus-visible,select:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
  .row2{display:grid;grid-template-columns:2fr 1fr;gap:10px}
  .two{display:grid;grid-template-columns:1fr 1fr;gap:14px}

  .pills{display:flex;gap:7px;flex-wrap:wrap}
  .pill{flex:1 1 0;min-width:42px;background:var(--canvas);border:1px solid var(--line);color:var(--muted);
    border-radius:10px;padding:11px 0;font-size:14px;font-weight:600;cursor:pointer;text-align:center;
    font-variant-numeric:tabular-nums;font-family:var(--sans);transition:border-color .12s,color .12s,background .12s}
  .pill:hover{border-color:var(--sea);color:var(--ink)}
  .pill[aria-pressed="true"]{background:var(--gold-soft);border-color:var(--gold);color:var(--gold)}
  .pill:focus-visible{outline:2px solid var(--gold);outline-offset:2px}

  .go{width:100%;margin-top:22px;border:none;cursor:pointer;background:var(--gold);color:#231708;
    font-size:16px;font-weight:700;font-family:var(--sans);padding:16px;border-radius:13px;letter-spacing:.01em;
    transition:transform .08s,filter .15s}
  .go:hover{filter:brightness(1.05)}
  .go:active{transform:translateY(1px)}
  .go:disabled{opacity:.6;cursor:wait}

  /* results */
  .result{margin-top:22px;background:var(--panel-hi);border:1px solid var(--line);border-radius:16px;padding:24px}
  .rlabel{font-size:11.5px;letter-spacing:.22em;text-transform:uppercase;color:var(--muted)}
  .center{text-align:center}
  .value{font-family:var(--serif);font-weight:600;font-size:50px;line-height:1.02;color:var(--gold);
    font-variant-numeric:tabular-nums;margin:8px 0 2px;letter-spacing:-.01em}
  .range{color:var(--ink);font-size:15px;margin-top:6px}
  .range b{color:var(--sea);font-weight:600;font-variant-numeric:tabular-nums}
  .chips{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:16px}
  .chip{background:var(--canvas);border:1px solid var(--line-soft);border-radius:999px;padding:7px 14px;
    font-size:13px;color:var(--ink);font-variant-numeric:tabular-nums}
  .chip b{font-weight:600}.chip.rate b{color:var(--gold)}
  .chip.up{border-color:rgba(232,176,75,.4)}.chip.up b{color:var(--gold)}
  .chip.down{border-color:rgba(95,182,168,.4)}.chip.down b{color:var(--sea)}

  .guide{margin-top:20px;border-top:1px solid var(--line);padding-top:16px}
  .guide .gl{font-size:12px;color:var(--muted);margin-bottom:10px}
  .guide .gl b{color:var(--ink)}
  .gbars{display:grid;gap:8px}
  .gbar{display:grid;grid-template-columns:46px 1fr auto;align-items:center;gap:10px;font-size:13px}
  .gbar .gk{color:var(--muted)}
  .gtrack{height:8px;background:var(--canvas);border-radius:5px;overflow:hidden}
  .gfill{height:100%;background:var(--sea);border-radius:5px}
  .gbar .gv{color:var(--ink);font-variant-numeric:tabular-nums}

  .emi{margin-top:20px;border-top:1px solid var(--line);padding-top:16px;text-align:center}
  .emiVal{font-weight:700;font-size:26px;color:var(--ink);font-variant-numeric:tabular-nums;margin-top:6px}
  .emiVal span{color:var(--muted);font-weight:500;font-size:14px}
  .tenure{justify-content:center;max-width:300px;margin:13px auto 0}
  .tenure .pill{padding:8px 0;font-size:13px;min-width:64px}
  .emiNote{color:var(--muted);font-size:12px;margin-top:10px}

  .foot{color:var(--muted);font-size:12px;margin-top:18px;text-align:center;line-height:1.5}
  .foot .dot{color:var(--line);margin:0 6px}
  .err{color:#ef8f7a;font-size:14px;text-align:center}

  .adjrow{display:flex;align-items:flex-start;gap:10px;margin-top:20px;cursor:pointer;
    font-size:12.5px;color:var(--muted);line-height:1.45}
  .adjrow input{width:17px;height:17px;margin-top:1px;accent-color:var(--gold);cursor:pointer;flex:0 0 auto}
  .adjrow b{color:var(--gold);font-weight:600}
  .adjtag{display:inline-block;margin-top:8px;font-size:11.5px;color:var(--sea)}

  .trend{margin-top:20px;border-top:1px solid var(--line);padding-top:16px}
  .trend .tl{font-size:12px;color:var(--muted);margin-bottom:4px}
  .trend .tl b{color:var(--ink)}
  .trend .tcap{font-size:12px;color:var(--sea);margin-top:8px}
  .trend svg{width:100%;height:80px;display:block;margin-top:8px}

  /* compare results */
  .cmpgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .ctile{background:var(--canvas);border:1px solid var(--line);border-radius:13px;padding:18px;text-align:center}
  .ctile .cl{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--sea);font-weight:700}
  .ctile .cv{font-family:var(--serif);font-weight:600;font-size:30px;color:var(--gold);font-variant-numeric:tabular-nums;margin:6px 0 2px}
  .ctile .cs{font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums}
  .verdict{margin-top:14px;text-align:center;font-size:15px;color:var(--ink)}
  .verdict b{color:var(--gold);font-weight:600}

  @media (max-width:620px){
    .homes.compare{grid-template-columns:1fr}
    .cmpgrid{grid-template-columns:1fr}
    h1{font-size:30px}.value{font-size:42px}
    body{padding:24px 16px 48px}
  }
  @media (prefers-reduced-motion:reduce){.go,.pill,.modebtn{transition:none}}
</style>
</head>
<body>
<div class="wrap">

  <div class="brand">
    <svg class="mark" viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <path d="M4 15 16 5l12 10" stroke="#e8b04b" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M7 14v12h18V14" stroke="#5fb6a8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M13 26v-7h6v7" stroke="#5fb6a8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span class="word">Ghar<b>IQ</b></span>
    <span class="tag">ghar ka smart estimate</span>
  </div>

  <h1>What's this home <em>really</em> worth?</h1>
  <p class="sub">Estimate a market value, see how it compares to the city, check a rough EMI &mdash; or put two homes side by side. Built from 84,000+ real listings.</p>

  <div class="card">
    <div class="modebar" role="tablist">
      <button class="modebtn" id="mSingle" role="tab" aria-selected="true">Single home</button>
      <button class="modebtn" id="mCompare" role="tab" aria-selected="false">Compare two</button>
    </div>

    <div class="homes" id="homes"></div>

    <label class="adjrow"><input type="checkbox" id="adjust" checked>
      <span>Adjust to <b>2025</b> prices &mdash; model learns from 2020&ndash;23 listings, so we index it up using RBI data</span></label>

    <button class="go" id="go">Estimate value</button>

    <div id="results"></div>
  </div>
</div>

<script>
  const CITIES=["Mumbai","Thane","Navi Mumbai","Pune","Nagpur","Nashik","New Delhi","Gurgaon","Noida","Greater Noida","Ghaziabad","Faridabad","Bangalore","Hyderabad","Chennai","Kolkata","Ahmedabad","Surat","Vadodara","Jaipur","Lucknow","Kanpur","Indore","Bhopal","Chandigarh","Mohali","Kochi","Coimbatore","Visakhapatnam","Patna","Bhubaneswar","Goa","Mangalore","Mysore","Dehradun"];
  const CITY_PPS={"Bangalore":6190,"Kolkata":5000,"Pune":7130,"Chennai":6130,"Jaipur":4000,"Gurgaon":10530,"Faridabad":5040,"Noida":5650,"Mumbai":15900,"Greater Noida":5600,"Vadodara":3750,"Ahmedabad":6880,"Hyderabad":6860,"Surat":4290,"New Delhi":12180,"Ghaziabad":4570,"Chandigarh":4580,"Mohali":4810,"Visakhapatnam":4830,"Kochi":5570,"Lucknow":5170,"Dehradun":5000,"Thane":11950,"Nagpur":4310,"Bhubaneswar":5370,"Mangalore":4530,"Goa":6470,"Indore":3300,"Navi Mumbai":10980,"Nashik":4860,"Mysore":5490,"Patna":6000,"Kanpur":5000,"Coimbatore":4990,"Bhopal":2930};
  const LOCALITIES={"Mumbai":["Thane West","Ghodbunder Road","Mulund (West)","Kolshet Road","Kharghar","Kandivali (East)"],"Pune":["Baner","Wagholi","Kharadi","Wakad","NIBM","Hadapsar"],"Nagpur":["Manish Nagar"],"Gurgaon":["Emaar Digi Homes","Sector-65 Gurgaon","M3M Golf Estate","Sector-102 Gurgaon","Ireo Victory Valley","Sector-81 Gurgaon"],"Noida":["Sector-137 Noida","Noida Extension","Sector-75 Noida","Sector-150 Noida","Sector-78 Noida","Sector-74 Noida"],"Greater Noida":["CRC Sublimis","Purvanchal Royal City","Arihant Ambar","Gaur Saundaryam","Sikka Kaamya Greens","Devsai Sportshome"],"Ghaziabad":["Raj Nagar Extension","Crossing Republik","Vaibhav Khand","Indirapuram","Ahinsa Khand 2","Bhopura"],"Faridabad":["Sector 88 Faridabad","Sector 86 Faridabad","RPS Savana","Sector 82 Faridabad","Honour Homes","Amolik Residency"],"Bangalore":["Whitefield","Thanisandra","Sarjapur Road","Electronics City Phase 1","Hebbal","Yelahanka"],"Chennai":["Porur","Guduvancheri","Medavakkam","Sholinganallur","Perumbakkam","Poonamallee"],"Kolkata":["New Town","Rajarhat","E M Bypass","Maheshtala","Garia","Madhyamgram"],"Ahmedabad":["Godrej Green Glades","Shaligram Prime","Swati Parkside","Orchid Legacy"],"Surat":["Palanpur","PAL","Vesu","Adajan","Althan"],"Vadodara":["Vasna-Bhayli-Road","Gotri","Manjalpur"],"Jaipur":["Jagatpura","Mansarovar","Ajmer Road","Vaishali Nagar","Mansarovar Extension","Gandhi Path"],"Lucknow":["Gomti Nagar Extension","Sushant Golf City","Faizabad Road"],"Indore":["Nipania","Bicholi Mardana"],"Chandigarh":["Zirakpur","Sushma Grande","Maxxus Elanza","Sushma Valencia","Uptown Skylla","Steel Strips Towers"],"Mohali":["Zirakpur","Ambika La Parisian","Sunny Enclave","Sector-88 Mohali","Sector-66A Mohali","Gillco Parkhills"],"Kochi":["Kakkanad","TATA Tritvam"],"Visakhapatnam":["Madhurawada","Novus Florence Village"],"Bhubaneswar":["Patia","Sundarpada"],"Dehradun":["Pacific Golf Estate","Windlass River Valley","Sikka Kimaya Greens"]};
  const PRICEGUIDE={"Mumbai":{"1":6500000,"2":13500000,"3":25000000},"Thane":{"1":4240000,"2":9700000,"3":19000000},"Navi Mumbai":{"1":5000000,"2":9800000,"3":16600000},"Pune":{"1":3300000,"2":6500000,"3":12000000},"Nagpur":{"1":2350000,"2":3900000,"3":6500000},"Nashik":{"1":2500000,"2":4350000,"3":6820000},"New Delhi":{"1":2800000,"2":7500000,"3":18000000},"Gurgaon":{"1":2200000,"2":8450000,"3":17500000},"Noida":{"1":2500000,"2":5500000,"3":9000000},"Greater Noida":{"1":1990000,"2":5100000,"3":8000000},"Ghaziabad":{"1":2000000,"2":4000000,"3":7000000},"Faridabad":{"1":1850000,"2":4200000,"3":7200000},"Bangalore":{"1":3950000,"2":6170000,"3":10000000},"Hyderabad":{"1":3300000,"2":6150000,"3":12600000},"Chennai":{"1":3000000,"2":5000000,"3":9000000},"Kolkata":{"1":2000000,"2":3500000,"3":7300000},"Ahmedabad":{"1":2500000,"2":5000000,"3":9000000},"Surat":{"1":2000000,"2":4200000,"3":7580000},"Vadodara":{"1":1700000,"2":3150000,"3":5500000},"Jaipur":{"1":1840000,"2":3500000,"3":5510000},"Lucknow":{"1":2180000,"2":5200000,"3":8260000},"Kanpur":{"1":2250000,"2":4200000,"3":7000000},"Indore":{"1":1700000,"2":3150000,"3":5800000},"Bhopal":{"2":2500000,"3":3550000},"Chandigarh":{"1":2400000,"2":4500000,"3":7000000},"Mohali":{"1":2000000,"2":4080000,"3":7000000},"Kochi":{"1":2850000,"2":5500000,"3":8500000},"Coimbatore":{"1":2850000,"2":5000000,"3":8000000},"Visakhapatnam":{"2":4200000,"3":8000000},"Patna":{"1":4000000,"2":5500000,"3":7670000},"Bhubaneswar":{"2":5200000,"3":8700000},"Goa":{"1":4100000,"2":6500000,"3":11000000},"Mangalore":{"1":2520000,"2":4500000,"3":7900000},"Mysore":{"2":6000000,"3":8000000},"Dehradun":{"1":3200000,"2":5200000,"3":8000000}};

  // RBI/BIS India residential property price index (yearly avg)
  const PRICEINDEX={2018:277,2019:286,2020:294,2021:301,2022:313,2023:324,2024:347,2025:360};
  const FACTOR_2025=PRICEINDEX[2025]/PRICEINDEX[2021];   // ~1.20 : 2020-23 data -> 2025 level
  const UPLIFT_PCT=Math.round((FACTOR_2025-1)*100);

  const $=id=>document.getElementById(id);
  const inr=n=>"\u20b9"+Math.round(n).toLocaleString("en-IN");
  function crLakh(n){ if(n>=1e7)return "\u20b9"+(n/1e7).toFixed(2)+" Cr"; if(n>=1e5)return "\u20b9"+(n/1e5).toFixed(1)+" Lakh"; return inr(n); }

  function buildPills(host,opts,def,onChange){
    let chosen=def;
    opts.forEach(o=>{
      const b=document.createElement("button");
      b.type="button";b.className="pill";b.textContent=o.label;
      b.setAttribute("aria-pressed",o.val===def?"true":"false");
      b.addEventListener("click",()=>{chosen=o.val;
        host.querySelectorAll(".pill").forEach(p=>p.setAttribute("aria-pressed","false"));
        b.setAttribute("aria-pressed","true"); if(onChange)onChange(chosen);});
      host.appendChild(b);
    });
    return ()=>chosen;
  }

  // build one home input block, return {el, get}
  function makeHome(title){
    const el=document.createElement("div"); el.className="home";
    const cityOpts=CITIES.map(c=>'<option>'+c+'</option>').join("");
    el.innerHTML=
      (title?'<div class="htitle">'+title+'</div>':'')+
      '<div class="two">'+
        '<div class="field"><label>City</label><select class="f-city">'+cityOpts+'</select></div>'+
        '<div class="field"><label>Locality <span class="opt">optional</span></label><select class="f-loc"></select></div>'+
      '</div>'+
      '<div class="field"><label>Built-up area</label><div class="row2">'+
        '<input type="number" class="f-area" value="1000" step="10" min="100">'+
        '<select class="f-unit"><option value="1">sq.ft</option><option value="9">gaz</option><option value="720">katha</option></select>'+
      '</div></div>'+
      '<div class="two">'+
        '<div class="field"><label>Bedrooms</label><div class="pills f-bhk"></div></div>'+
        '<div class="field"><label>Bathrooms</label><div class="pills f-bath"></div></div>'+
      '</div>';
    const citySel=el.querySelector(".f-city"), locSel=el.querySelector(".f-loc");
    function fillLoc(){
      const list=LOCALITIES[citySel.value]||[];
      locSel.innerHTML='<option value="Unknown">Any area</option>'+list.map(l=>'<option>'+l+'</option>').join("");
    }
    citySel.addEventListener("change",fillLoc); fillLoc();
    const getBhk=buildPills(el.querySelector(".f-bhk"),[{label:"1",val:1},{label:"2",val:2},{label:"3",val:3},{label:"4",val:4},{label:"5+",val:5}],2);
    const getBath=buildPills(el.querySelector(".f-bath"),[{label:"1",val:1},{label:"2",val:2},{label:"3",val:3},{label:"4+",val:4}],2);
    return {el, get(){
      const unit=parseFloat(el.querySelector(".f-unit").value);
      const areaSqft=parseFloat(el.querySelector(".f-area").value)*unit;
      return {city:citySel.value, locality:locSel.value||"Unknown", bhk:getBhk(), bathrooms:getBath(), area_sqft:areaSqft, _area:areaSqft};
    }};
  }

  // ---- mode ----
  let mode="single", homeA, homeB, lastSingle=null, getTenure=null;
  const homesWrap=$("homes");
  function renderMode(){
    homesWrap.innerHTML=""; homesWrap.classList.toggle("compare",mode==="compare");
    $("results").innerHTML="";
    homeA=makeHome(mode==="compare"?"Home A":""); homesWrap.appendChild(homeA.el);
    if(mode==="compare"){ homeB=makeHome("Home B"); homesWrap.appendChild(homeB.el); }
    $("mSingle").setAttribute("aria-selected",mode==="single");
    $("mCompare").setAttribute("aria-selected",mode==="compare");
    $("go").textContent=mode==="compare"?"Compare homes":"Estimate value";
  }
  $("mSingle").addEventListener("click",()=>{mode="single";renderMode();});
  $("mCompare").addEventListener("click",()=>{mode="compare";renderMode();});
  renderMode();

  async function predict(body){
    const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    if(!r.ok) throw new Error(r.status);
    return (await r.json()).predicted_price_inr;
  }
  function cmpChip(price,area,city){
    const med=CITY_PPS[city]; if(!med) return "";
    const diff=Math.round((price/area/med-1)*100);
    if(diff>=3) return '<span class="chip up"><b>'+diff+'% above</b> '+city+' avg</span>';
    if(diff<=-3) return '<span class="chip down"><b>'+Math.abs(diff)+'% below</b> '+city+' avg</span>';
    return '<span class="chip">around '+city+' avg</span>';
  }
  function guideHtml(city){
    const g=PRICEGUIDE[city]; if(!g) return "";
    const keys=Object.keys(g); const max=Math.max(...keys.map(k=>g[k]));
    const bars=keys.map(k=>{
      const w=Math.round(g[k]/max*100);
      return '<div class="gbar"><span class="gk">'+k+' BHK</span><div class="gtrack"><div class="gfill" style="width:'+w+'%"></div></div><span class="gv">'+crLakh(g[k])+'</span></div>';
    }).join("");
    return '<div class="guide"><div class="gl">Typical prices in <b>'+city+'</b> (median, from listings)</div><div class="gbars">'+bars+'</div></div>';
  }
  function emiVal(price,years){
    const loan=price*0.8, r=0.085/12, n=years*12;
    return loan*r*Math.pow(1+r,n)/(Math.pow(1+r,n)-1);
  }

  function trendHtml(){
    const yrs=Object.keys(PRICEINDEX).map(Number), vals=yrs.map(y=>PRICEINDEX[y]);
    const mn=Math.min(...vals), mx=Math.max(...vals), W=320,H=64,pad=6;
    const xs=yrs.map((y,i)=>pad+i*((W-2*pad)/(yrs.length-1)));
    const ys=vals.map(v=>H-pad-((v-mn)/(mx-mn))*(H-2*pad));
    const pts=xs.map((x,i)=>x.toFixed(1)+','+ys[i].toFixed(1)).join(' ');
    let area='M'+xs[0].toFixed(1)+','+(H-pad);
    xs.forEach((x,i)=>{ area+=' L'+x.toFixed(1)+','+ys[i].toFixed(1); });
    area+=' L'+xs[xs.length-1].toFixed(1)+','+(H-pad)+' Z';
    return '<div class="trend"><div class="tl">India home-price trend <b>(RBI/BIS index)</b></div>'+
      '<svg viewBox="0 0 '+W+' '+(H+14)+'" preserveAspectRatio="none" aria-hidden="true">'+
        '<path d="'+area+'" fill="rgba(95,182,168,.12)" stroke="none"/>'+
        '<polyline points="'+pts+'" fill="none" stroke="#5fb6a8" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'+
        '<circle cx="'+xs[xs.length-1].toFixed(1)+'" cy="'+ys[ys.length-1].toFixed(1)+'" r="3" fill="#e8b04b"/>'+
        '<text x="'+pad+'" y="'+(H+11)+'" fill="#88a59f" font-size="9">'+yrs[0]+'</text>'+
        '<text x="'+(W-pad)+'" y="'+(H+11)+'" fill="#88a59f" font-size="9" text-anchor="end">'+yrs[yrs.length-1]+'</text>'+
      '</svg>'+
      '<div class="tcap">Up ~'+UPLIFT_PCT+'% since 2021 \u00b7 ~'+Math.round((mx/vals[0]-1)*100)+'% since 2018</div></div>';
  }

  function renderSingle(price,h,adjusted){
    lastSingle={price, h};
    const tag = adjusted
      ? '<div class="adjtag">Indexed to 2025 prices (+'+UPLIFT_PCT+'%) using RBI data</div>'
      : '<div class="adjtag">As-listed level (2020\u201323 data)</div>';
    const foot = adjusted
      ? 'Model learns from 2020\u201323 listings, indexed to 2025 via RBI data<span class="dot">\u00b7</span>a ballpark, not a valuation'
      : 'Estimates from 2020\u201323 listings<span class="dot">\u00b7</span>a ballpark, not a valuation \u2014 floor, age &amp; condition also matter';
    $("results").innerHTML=
      '<div class="result center">'+
        '<div class="rlabel">Estimated market value</div>'+
        '<div class="value" id="val">'+crLakh(price)+'</div>'+
        tag+
        '<div class="range">Likely range <b>'+crLakh(price*0.88)+'</b> \u2013 <b>'+crLakh(price*1.12)+'</b></div>'+
        '<div class="chips"><span class="chip rate">'+inr(price/h._area)+' <b>/ sq.ft</b></span>'+cmpChip(price,h._area,h.city)+'</div>'+
        guideHtml(h.city)+
        trendHtml()+
        '<div class="emi"><div class="rlabel">If you take a home loan</div><div class="emiVal" id="emiVal">'+inr(emiVal(price,20))+' <span>/ month</span></div>'+
          '<div class="pills tenure" id="tenure"></div><div class="emiNote">Assuming 20% down payment \u00b7 8.5% interest</div></div>'+
        '<div class="foot">'+foot+'</div>'+
      '</div>';
    getTenure=buildPills($("tenure"),[{label:"15 yr",val:15},{label:"20 yr",val:20},{label:"25 yr",val:25}],20,
      v=>{ $("emiVal").innerHTML=inr(emiVal(price,v))+' <span>/ month</span>'; });
  }

  function renderCompare(a,b,adjusted){
    let verdict;
    const hi=Math.max(a.price,b.price), lo=Math.min(a.price,b.price);
    const pct=Math.round((hi/lo-1)*100), diff=hi-lo;
    if(pct<3) verdict='Both are priced <b>about the same</b>.';
    else{ const who=a.price>b.price?"Home A":"Home B"; verdict=who+' is <b>'+crLakh(diff)+'</b> ('+pct+'%) more expensive.'; }
    function tile(label,r,h){
      return '<div class="ctile"><div class="cl">'+label+'</div><div class="cv">'+crLakh(r)+'</div>'+
        '<div class="cs">'+inr(r/h._area)+' / sq.ft \u00b7 '+h.bhk+' BHK \u00b7 '+h.city+'</div></div>';
    }
    const foot = adjusted ? 'Indexed to 2025 prices (+'+UPLIFT_PCT+'%) using RBI data \u00b7 ballpark only'
                          : 'Estimates from 2020\u201323 listings \u00b7 ballpark only';
    $("results").innerHTML=
      '<div class="result"><div class="cmpgrid">'+tile("Home A",a.price,a.h)+tile("Home B",b.price,b.h)+'</div>'+
      '<div class="verdict">'+verdict+'</div>'+
      '<div class="foot">'+foot+'</div></div>';
  }

  async function run(){
    const btn=$("go"); btn.disabled=true; const old=btn.textContent; btn.textContent="Estimating\u2026";
    const factor=($("adjust")&&$("adjust").checked)?FACTOR_2025:1, adj=factor!==1;
    $("results").innerHTML='<div class="result center"><div class="foot" style="margin:0">Reading the market\u2026</div></div>';
    try{
      if(mode==="single"){
        const h=homeA.get(); const p=await predict(h); renderSingle(p*factor,h,adj);
      }else{
        const ha=homeA.get(), hb=homeB.get();
        const [pa,pb]=await Promise.all([predict(ha),predict(hb)]);
        renderCompare({price:pa*factor,h:ha},{price:pb*factor,h:hb},adj);
      }
    }catch(e){
      $("results").innerHTML='<div class="result"><div class="err">Couldn\u2019t reach the model \u2014 make sure the server is running, then try again.</div></div>';
    }finally{ btn.disabled=false; btn.textContent=old; }
  }
  $("go").addEventListener("click",run);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE