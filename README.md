# Praktikum strojnega učenja v fiziki

[Povezava](https://ucilnica.fmf.uni-lj.si/course/view.php?id=698) do spletne učilnice predmeta.

## Modeliranje 1-D porazdelitve: razpadi Higgsovega bozona $`H\rightarrow\mu\mu`$

### Navodila in usmeritve

V nadaljevanju sledijo podrobnejša navodila in usmeritve za reševanje naloge.

#### 1. del

1. Iz surovih ("raw") podatkov zgeneriraj svoje histograme (priporočeno!) s pomočjo predpripravljene skripte `create_histograms.py`, pri kateri lahko spreminjaš število predalov ("bin"-ov) in $`m_{\mu\mu}`$ interval, ki ga boš opazoval/-a. Histogrami (mejne in sredinske $`x`$ vrednosti predalov, vrednosti in napake) se shranijo v formatu `.npz`. Na voljo imaš že nekaj generiranih histogramov v `data/original_histograms/`, ki jih lahko uporabiš namesto generacije novih histogramov in nalaganja podatkov.

2. Ko imaš zgenerirane svoje histograme (ali pa uporabiš že narejene), jih lahko izrišeš s pomočjo skripte `visualize_data.py` (ustrezno s prejšnjo točko spremeni ime datotek, ki jih nalagaš).

3. Preveri, če so napake res pravilno upoštevane. Lahko jih namenoma pokvariš in ponoviš prva dva koraka, da vidiš vpliv.

4. Da se spoznaš z osnovnim fitanjem, najprej zgladi histogram simuliranega ozadja ("simulated background") s pomočjo preprostejših matematičnih funkcij in nadaljuj do različnih teoretično podkrepljenih nastavkov (CMS, ATLAS nastavki). Dobiš funkcijo/vrednosti predalov $`m(x_k)`$. Preizkusi primere iz vaj in uporabi `atlas_fit_function.py`.

5. Prilagodi funkcijo CB histogramu simuliranega signala, pri čemer upoštevaj še dodatni normalizacijski faktor. Dobiš funkcijo/vrednosti predalov $`s(x_k)`$. Primer je na voljo v `fit_CB.py`.

#### 2. del

6. Ker simulacija ozadja ni vedno najboljša, se po navadi za oceno ozadja raje vzame izmerjene podatke, pri čemer pa je potrebno izključiti območje, kjer pričakujemo signal ("blinding") - nočemo fitati še signala! Prilagodi torej funkcijo histogramu podatkov, da dobiš dobro oceno za ozadje ("background from data") in pri tem pazi, da pri fitu **ne** upoštevaš območja okrog mase Higgsovega bozona, npr. izključi interval 120 - 130 GeV. Dobiš funkcijo/vrednosti predalov $`b(x_k)`$. V tem koraku preizkusi tudi ML metode regresije (KRR, SVR, GPR, ...) za fitanje ozadja iz podatkov. Pomagaš si lahko s primeri v `fit_GPR_{simple, smooth, logarithm}.py`.

8. Od podatkov odštej čim bolje zglajeno ozadje, ki si ga dobil/-a v prejšnji točki, da dobiš ekstrahiran signal. Če so vrednosti podatkov $`d(x_k)`$, dobimo ekstrahiran signal $`y(x_k)`$ kot $`y(x_k) = d(x_k) - b(x_k)`$.

9. Na ekstrahiran signal fitaj CB funkcijo s prostimi parametri, ki si jih dobil/-a v točki 5 tako, da ji v resnici prilagodiš le nov normalizacijski faktor, npr.: $`\alpha \cdot s(x_k)`$. Optimalno je, da je le-ta blizu 1.

10. Ker je izmerjenega signala še zelo malo, predlagamo, da postopek najprej narediš z umetno napihnjenim signalom - le  tega množi z nekim faktorjem (npr. $`\gamma = 100`$) in ga dodaj podatkom: $`s_\textrm{new}(x_k) = \gamma \cdot s(x_k)`$ in $`d(x_k) = d(x_k) + s_\textrm{new}(x_k)`$. Ker bo signal na ta način lepo izstopal iz ozadja, ga boš lažje izluščil/-a. Primer napihnjenega signala je v skripti `create_asimov.py`.

### Pridobitev kode iz repozitorija

```shell
git clone https://github.com/j-gavran/PSUF_Hmumu.git
```

### Postavitev virtualnega okolja

Virtualno okolje se naloži preko pip-a:

```shell
pip install pip --upgrade
pip install virtualenv
```

in se ga postavi z ukazom:
```shell
python -m venv .venv
```

kjer `.venv` predstavlja ime virtualnega okolja. To okolje se aktivira z ukazom:

```shell
source .venv/bin/activate
```

Za izhod iz okolja se uporabi ukaz:

```shell
deactivate
```

### Namestitev knjižnic

Knjižnice se namestijo z ukazom:

```shell
pip install -r requirements.txt
```

Probleme s Python path importi se reši z ukazom iz terminala (v direktoriju tega repozitorija):

```shell
export PYTHONPATH=.
```
