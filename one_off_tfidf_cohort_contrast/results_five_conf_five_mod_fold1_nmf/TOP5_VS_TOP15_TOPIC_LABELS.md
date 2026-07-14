# Five versus fifteen terms per NMF topic

The NMF fit is unchanged in this comparison. Both labeling batches use the
same 100 topics; only the number of highest-loading terms shown to the Gemma
agent differs.

| Metric | 15 terms | 5 terms |
|---|---:|---:|
| Topic responses | 100 | 100 |
| Specific features returned | 376 | 172 |
| Unique exact feature names | 274 | 149 |
| Structured candidates returned | 375 | 196 |
| Unique exact candidate names | 287 | 170 |
| Mean features per topic | 3.76 | 1.72 |
| Topics with no specific features | 0 | 8 |
| Topics called coherent | 52 | 65 |
| Topics called mixed | 44 | 23 |
| Topics called mostly artifact | 4 | 12 |

The five-term run retained the same broad hidden-modifier recovery:

- brain metastases was identified directly;
- histology was identified, but the main topic lost its explicit squamous
  subtype because `squamous histology.` was the sixth-loading term;
- EGFR status was identified, but KRAS disappeared from topic rank 50 because
  its phrases were seventh and eighth;
- hemoglobin was identified more cleanly as a stand-alone measurement;
- NLR was still not identified, although a WBC/ANC topic remained.

Thus, five terms reduce feature proliferation substantially without losing a
broad modifier family in this fold, but they omit clinically useful details
that occur below the fifth loading. The exact-name overlap between runs was
low because the model also normalized and named concepts differently when
given less context; raw feature counts should not be interpreted as unique
ontology concepts without a second cross-topic consolidation step.
