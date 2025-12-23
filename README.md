# Guida al Preprocessing di Volumi NIFTI

Questo documento descrive i passi necessari per preprocessare i volumi NIfTI e prepararli per lo skull-stripping e la normalizzazione.

---

## Requisiti della pipeline

### Risoluzione

- Tutti i volumi devono avere risoluzione isotropica (1, 1, 1) mm.
- I volumi che non rispettano questo requisito devono essere ricampionati.

### Struttura del dataset accettata

La pipeline supporta due strutture standard.
'''
Dataset/
└── sub-X/
└── ses-X/
└── anat/
└── <file.nii.gz>
'''
oppure:
'''
Dataset/
└── sub-X/
└── anat/
└── <file.nii.gz>
'''

La pipeline supporta anche dataset misti: alcuni soggetti possono avere la cartella `ses-X` e altri solamente `anat`.

Se il dataset non segue una di queste strutture, è necessario creare uno script per convertirlo nel formato corretto.

---

## Passaggi di Preprocessing

# 1. Analisi dei volumi

Scrivere un codice Python che:

- Analizza tutti i volumi presenti nel dataset.
- Estrae shape, voxel size, affine, datatype e altre informazioni utili.
- Identifica i volumi che non hanno risoluzione (1,1,1).
- Salva in un file Excel:
  - Il percorso completo del file
  - Le dimensioni voxel
  - La shape
  - L’affine
  - Una nota (OK / NON 1mm)
- Se necessario distinguere soggetti sani e non sani, utilizzare le liste contenute nei dataset originali (ad esempio su OpenNeuro).

---

# 2. Ricampionamento (Resampling)
Se i volumi non hanno le stesse risoluzioni, far partire il codice $ Resampling.py $. La struttura da dare in input è la seguente:
'''
Dataset
'''
All'interno del dataset la struttura deve essere come quelle definite in precedenza.

### Struttura di output

Lo script crea automaticamente la cartella `preproc_out` all’interno della cartella `anat` e salva i file ricampionati:
'''
Dataset/
└── sub-X/
└── ses-X/
└── anat/
├── sub.nii.gz
└── preproc_out/
├──sub..._mod_1mm.nii.gz
'''
dove mod è la modalità del soggetto preso in input.

### 3. Coregistrazione
Usa il codice `Coregistration.py` (adattandolo alla struttura del dataset):
- Per ogni soggetto, prende la **T1** come riferimento.
- Coregistra a essa tutti gli altri volumi con modalità diverse.
- Questo passaggio è fondamentale per permettere lo **skull-stripping** successivo.
La struttura dei dati in input deve essere quella raccomandata dall'output dell'operazione di resampling.
L'output prodotto è un dataset nuovo che contiene la seguente struttura:
'''
E:\Datasets\VOLUMI-SANI-1mm_coregistrati\
└── sub-01\
    └── anat\
        └── coregistrati_alla_t1\
            ├── sub-01_T1w.nii.gz
            ├── sub-01_T2w.nii.gz
            ├── sub-01_FLAIR.nii.gz
            ├── sub-01_PD.nii.gz
            ├── sub-01_T2w_to_T1.tfm
            ├── sub-01_FLAIR_to_T1.tfm
            └── sub-01_PD_to_T1.tfm
'''
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
### Comandi disponibili per `skullstripping.py`

Lo script `skullstripping.py` si usa da riga di comando e accetta i seguenti argomenti:

```bash
python skullstripping.py --robex_dir <PERCORSO_ROBEX> --subjects_root <PERCORSO_SOGGETTI> [--seed N] [--outside_value V]
```
--robex_dir indica la cartella dove si trova il file robex.exe
--subjects_root devi dare il percorso del dataset che vuoi skullstrippare. La struttura deve essere come quella che restituisce in output coregistration.py
--seed per avere gli stessi risultati
--outside_value per imporre a 0 in caso i valori al di fuori del volume (inutile)
#### ⚙️ Dipendenza necessaria: ROBEX-V12
Per eseguire correttamente lo skull-stripping è necessario scaricare **ROBEX-V12**, uno strumento esterno utilizzato dal codice:

🔗 [Scarica ROBEX-V12 da NITRC](https://www.nitrc.org/projects/robex)
***Importante*** quando esegui skulltripping.py assicurati che in --robex_dir inserisci il path della cartella che contiene il file .bat  (../ROBEX)
---

### 5. Registrazione allo standard MNI152
Usa `registration_MNI152_VOLUMI_SANI.py`:
- Per ogni soggetto, prende la T1 e la coregistra allo standard MNI152.
- Propaga la trasformazione a tutte le altre modalità del soggetto.
- Questo passaggio produce volumi uniformemente registrati nello spazio standard.
- La cartella di input deve essere quella di output prodotto dal codice skullstripping.py
- L'output della struttura è il seguente:
'''
E:\Datasets\VOLUMI-SANI-1mm_coregistrati\
└── sub-01\
    └── anat\
        └── volumi_coregistrati_alla_t1\
            ├── T1w_stripped_to_mni.nii.gz
            ├── FLAIR_stripped_to_mni.nii.gz
            └── T2w_stripped_to_mni.nii.gz
'''
Viene generato un nuovo dataset con i volumi coregistrati all'mni152
---
Per la sezione volumi malati, l'input folder deve essere della seguente struttura:
'''
ds005752-download_coregistrati/
└── sub-001/
    └── anat/
        └── skullstripped/
            ├── <T1.nii(.gz)>
            ├── <FLAIR.nii(.gz)>   
            ├── <T2.nii(.gz)>     
            └── altri file...
'''
I file devono avere le modalità in modo esplicito
# BIAS FIELD CORRECTION  
Utilizzando come input la cartella di output del codice della registrazione dei volumi sani, utilizza il codice $n4biasfield_sitk.py$
L'output sarà il seguente:
'''
C:\Users\Stefano\Desktop\Stefano\Datasets\NihmBias\
└── sub-001\
    └── anat\
        └── anat_bias\
            ├── sub-001-T1-bias.nii.gz
            ├── sub-001-T2-bias.nii.gz
            └── sub-001-FLAIR-bias.nii.gz
'''

# ESTRAZIONE DELLE IMMAGINI ASSIALI
Utilizzando la struttura che ha restituito il codice del bias, utilizzalo sull'estrazione delle immagini 
L'output sarà:
'''
path addestramento e testing
└── train\
    └── trainA
    └── trainB
└── test\
    └── SubX
        └── test
        └── anat
'''

 # ADDESTRAMENTO
 Per fare l'addestramento utilizzare il codice che si trova nel path:
 '''
 C:\Users\Stefano\Desktop\Stefano\CycleGan\pytorch-CycleGAN-and-pix2pix
 '''
 python train.py --dataroot C:\Users\Stefano\Desktop\Stefano\CycleGan\pytorch-CycleGAN-and-pix2pix\data_training\trainT1_FLAIR_SANI\train --name from_t1_to_flair_viceversa_SANI --model cycle_gan --no_flip --preprocess none --n_epochs 100 --num_threads 0 --gpu_ids 1 --display_id -1 --batch_size 1 --input_nc 1 --output_nc 1 --checkpoints_dir C:\Users\Stefano\Desktop\Stefano\CycleGan\pytorch-CycleGAN-and-pix2pix\checkpoints\CYCLE_T1_FLAIR_SANI

Per il testing singolo fare:
python test.py --dataroot C:\Users\Rosini\Desktop\ProgettoSTEFANONEW\Progetto\ConMatteo\MSSEG-Testing\Testing\Center_07\Patient_07\Preprocessed_Data\Output\test --name from_t1_to_t2_viceversa --model cycle_gan --results_dir C:\Users\Rosini\Desktop\ProgettoSTEFANONEW\Progetto\ConMatteo\MSSEG-Testing\Testing\Center_07\Patient_07\Preprocessed_Data\Output\GENERATED --input_nc 1 --output_nc 1 --preprocess none --no_flip --num_test 500
Fare riferimento al file txt $opzioni_addestramento_testing$

# Pipeline generazione volumi:
runnare il codice main pipeline e mettere come dati di input la cartella di testing.
Il codice nella tessa cartella genererà delle varie sottocartelle aventi tutti i passaggi della geneerazione del volume.

FINE.

E successivamente per l'addestramento fare la seguente operazione:
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

SE VE LA SENTITE, MIGLIORATE LA LOGICA DEL CODICE E CERCATE DI RENDERLO PIU' EFFICIENTE, ANCHE SE NON SERVE
SKULLSTRIPPING.PY FUNZIONA ANCHE CON ALTRI TOOL DI SKULLSTRIPPING, SE NON VI PIACE ROBEX CAMBIATE TOOL. IL TOOL CHE CAMBIERETE DEVE AVERE I SUOI OPPORTUNI COMANDI BASH PER ESSERE ESEGUITO E MODIFICATE IL CODICE CAMBIANDO IL BASH
---

