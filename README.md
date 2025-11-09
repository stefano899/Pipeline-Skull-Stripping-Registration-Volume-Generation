# Guida al Preprocessing di Volumi NIFTI 

Questo documento descrive i passi necessari per preprocessare i volumi e prepararli per lo skull-stripping e la normalizzazione.

---

## 🧠 Struttura del progetto
All’interno del dataset è presente una cartella con il codice per effettuare lo **skull-stripping** e i vari passi di **preprocessing**:

```
E:\Datasets\Volumi_sani_1mm_MNI\SkullStripping
```

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
```
> Usa questo comando per vedere tutti i parametri richiesti in input.  
> Assicurati di trovarti nella stessa directory del file `skullstripping.py` quando lo esegui.

#### ⚙️ Dipendenza necessaria: ROBEX-V12
Per eseguire correttamente lo skull-stripping è necessario scaricare **ROBEX-V12**, uno strumento esterno utilizzato dal codice:

🔗 [Scarica ROBEX-V12 da NITRC](https://www.nitrc.org/projects/robex)

Una volta scaricato, assicurati che il percorso all’eseguibile di ROBEX sia configurato correttamente nel tuo script o nel PATH di sistema.

---

### 5. Registrazione allo standard MNI152
Usa `registration_MNI152_VOLUMI_SANI.py`:
- Per ogni soggetto, prende la T1 e la coregistra allo standard MNI152.
- Propaga la trasformazione a tutte le altre modalità del soggetto.
- Questo passaggio produce volumi uniformemente registrati nello spazio standard.

---

## ⚠️ Avvertenze importanti

### 🔒 Non modificare la logica del codice
È **fondamentale** non modificare la logica interna dei seguenti script:
- `Resampling.py`
- `Coregistration.py`
- `skullstripping.py`
- `registration_MNI152_VOLUMI_SANI.py`

Questi codici implementano passaggi essenziali e delicati del preprocessing, e qualsiasi modifica alla loro logica potrebbe compromettere la coerenza e la riproducibilità dei risultati.

---

### 🧩 Cosa puoi modificare
Puoi modificare **solo** la parte relativa alla struttura dei file, cioè:
- Come i file vengono letti in input.
- Come sono organizzate le directory.
- Come vengono passati i percorsi alle funzioni.

In altre parole, puoi adattare lo script alla tua struttura di dataset, ma **non** alterare il modo in cui il codice elabora i dati.

---

### 🤖 Fatti aiutare da ChatGPT
Se non sei sicuro su come adattare la struttura dei file o i percorsi di input,  
**fatti aiutare da ChatGPT**: può fornirti suggerimenti passo-passo per modificare correttamente i percorsi, i nomi dei file o i pattern di ricerca, mantenendo intatta la logica originale del codice.
Questo README è stato corretto da chatGPT, faccio schifo a scrivere
---

