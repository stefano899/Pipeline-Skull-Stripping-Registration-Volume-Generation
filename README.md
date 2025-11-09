# Auto-SkullStripping
# Guida al Preprocessing dei Volumi Sani (1mm MNI)

Questo documento descrive i passi necessari per preprocessare i volumi e prepararli per lo skull-stripping e la normalizzazione.

---

## 🧠 Struttura del progetto
All’interno del dataset è presente una cartella con il codice per effettuare lo **skull-stripping** e i vari passi di **preprocessing**:


---

## ⚙️ Passaggi di Preprocessing

### 1. Analisi dei volumi
Scrivi un codice Python che:
- Analizza tutti i volumi e identifica, per ogni soggetto, i volumi che **non hanno risoluzione (1,1,1)**.
- Salva il **path** dei file non conformi in un file Excel.
- Estragga tutte le informazioni possibili sui volumi.
- Se serve distinguere soggetti sani e non sani, consulta le liste nei dataset originali su OpenNeuro (di solito etichettati come *healthy control* o *patient*).

---

### 2. Ricampionamento (Resampling)
Utilizza il codice `Resampling.py` (adattandolo alla struttura del dataset):
- Legge dal file Excel i file che non hanno risoluzione (1,1,1).
- Li ricampiona automaticamente a questa risoluzione standard.

---

### 3. Coregistrazione
Usa il codice `Coregistration.py` (adattandolo alla struttura del dataset):
- Per ogni soggetto, prende la **T1** come riferimento.
- Coregistra a essa tutti gli altri volumi con modalità diverse.
- Questo passaggio è fondamentale per permettere lo **skull-stripping** successivo.

---

### 4. Skull-Stripping
Utilizza `skullstripping.py`:
- Esegue lo skull-stripping sulla T1 di ogni soggetto.
- Propaga la maschera calcolata anche alle altre modalità coregistrate, ottenendo volumi privi di teschio.
- Funziona tramite **riga di comando (cmd)**, quindi fornisci i parametri direttamente dal terminale.

Esempio di comando:
```bash
python skullstripping.py -h
