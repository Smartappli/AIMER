/**
 * Browser-native speech input and output for the authenticated RAG assistant.
 */
'use strict';

const VOICE_ASSISTANT_LANGUAGES = {
  fr: 'fr-FR',
  en: 'en-US',
  nl: 'nl-NL',
  de: 'de-DE'
};

const VOICE_ASSISTANT_TRANSLATIONS = {
  fr: {
    avatarAria: 'Avatar 3D animé de Giovani',
    avatarIdle: 'Disponible',
    avatarListening: 'Je vous écoute',
    avatarThinking: 'Je recherche',
    avatarSpeaking: 'Je vous réponds',
    avatarSuccess: 'Réponse terminée',
    avatarError: 'Une erreur est survenue',
    avatarBadge: 'Avatar 3D vocal',
    introSubtitle: 'Interrogez le corpus par écrit ou à la voix.',
    introDescription:
      'L’avatar vous écoute, recherche les références pertinentes et anime ses lèvres pendant la lecture de la réponse.',
    languageLabel: 'Langue de l’assistant',
    languageHelp: 'Ce choix règle la dictée, la voix et les réponses.',
    rateLabel: 'Vitesse de lecture',
    rateHelp: 'Le débit de la voix et des lèvres reste synchronisé.',
    rateChanged: 'Vitesse de lecture définie sur {rate}×.',
    conversationLabel: 'Conversation avec Giovani',
    greetingTitle: 'Bonjour, je suis Giovani. Comment puis-je vous aider ?',
    greetingBody:
      'Décrivez votre tâche, votre modalité ou vos contraintes pour recevoir des recommandations issues du corpus scientifique.',
    questionLabel: 'Votre question',
    queryPlaceholder: 'Ex. Quel modèle recommandez-vous pour classifier des images IRM ?',
    dictate: 'Dicter',
    stopDictation: 'Arrêter la dictée',
    send: 'Envoyer',
    searchingButton: 'Recherche…',
    readResponse: 'Lire la réponse',
    stop: 'Arrêter',
    autoRead: 'Lecture automatique',
    ready: 'Prêt à écouter votre question.',
    privacy:
      'AIMER ne conserve aucun enregistrement audio. La reconnaissance vocale peut être traitée par le service de votre navigateur selon sa propre politique de confidentialité.',
    noRecommendation:
      'Le corpus ne contient pas encore assez d’éléments pour formuler une recommandation.',
    recommendationsOne: '1 recommandation trouvée.',
    recommendationsMany: '{count} recommandations trouvées.',
    model: 'Modèle {index}',
    confidence: 'Confiance {value} %',
    source: 'Source : {value}. ',
    safety: 'Note de sécurité.',
    speechStopped: 'Lecture arrêtée.',
    speechInProgress: 'Lecture de la réponse en cours…',
    speechFinished: 'Lecture terminée.',
    speechFailed: 'La lecture vocale a échoué. Vous pouvez lire la réponse à l’écran.',
    audioCapture: 'Aucun microphone utilisable n’a été détecté.',
    notAllowed: 'L’accès au microphone a été refusé. Autorisez-le dans le navigateur.',
    noSpeech: 'Aucune parole n’a été détectée. Vous pouvez réessayer.',
    recognitionNetwork: 'Le service de reconnaissance vocale est indisponible.',
    recognitionAborted: 'La dictée a été arrêtée.',
    recognitionFailed: 'La dictée vocale a rencontré une erreur.',
    listening: 'Écoute en cours… Parlez maintenant.',
    dictationFinished: 'Dictée terminée. Vérifiez le texte puis envoyez votre question.',
    dictationStartFailed:
      'Impossible de démarrer la dictée. Attendez un instant puis réessayez.',
    dictationUnsupported: 'Dictée non prise en charge par ce navigateur',
    unsupported:
      'Certaines fonctions vocales ne sont pas prises en charge par ce navigateur. La saisie et la lecture à l’écran restent disponibles.',
    missingQuery: 'Saisissez ou dictez une question avant de l’envoyer.',
    corpusSearching: 'Recherche dans le corpus scientifique…',
    sessionExpired: 'Votre session a expiré. Reconnectez-vous pour utiliser l’assistant.',
    rateLimited: 'Trop de demandes ont été envoyées. Patientez une minute puis réessayez.',
    serviceUnavailable: 'Le service de recommandation est momentanément indisponible.',
    requestFailed: 'La demande n’a pas pu être traitée.',
    unexpectedError: 'Une erreur inattendue empêche l’assistant de répondre.',
    responseReceived: 'Réponse reçue.',
    demoStart: 'Lancer la démo vocale',
    demoStop: 'Arrêter la démo',
    demoHelp:
      'Découvrez la dictée, l’analyse RAG et la réponse parlée en un seul parcours.',
    demoStepDictation: 'Dictée',
    demoStepAnalysis: 'Analyse',
    demoStepResponse: 'Réponse',
    demoIntro:
      'Bonjour, je suis Giovani. Je vais démontrer mes capacités vocales. Après cette phrase, posez votre question à voix haute. Je la transcrirai, analyserai le corpus scientifique, puis lirai ma réponse.',
    demoListening: 'À vous de parler : posez maintenant votre question à Giovani.',
    demoComplete: 'Démonstration terminée. Giovani a écouté, analysé et répondu.',
    demoStopped: 'Démonstration arrêtée.',
    demoUnsupported:
      'La démonstration nécessite la dictée et la lecture vocales dans ce navigateur.',
    languageChanged: 'Langue de l’assistant définie sur le français.'
  },
  en: {
    avatarAria: 'Animated 3D avatar of Giovani',
    avatarIdle: 'Available',
    avatarListening: 'I’m listening',
    avatarThinking: 'Searching',
    avatarSpeaking: 'Speaking',
    avatarSuccess: 'Answer complete',
    avatarError: 'Something went wrong',
    avatarBadge: '3D voice avatar',
    introSubtitle: 'Ask the corpus by text or voice.',
    introDescription:
      'The avatar listens, searches for relevant references and moves its lips while reading the response.',
    languageLabel: 'Assistant language',
    languageHelp: 'This selection controls dictation, voice and responses.',
    rateLabel: 'Reading speed',
    rateHelp: 'Voice and lip movement stay synchronised at this speed.',
    rateChanged: 'Reading speed set to {rate}×.',
    conversationLabel: 'Conversation with Giovani',
    greetingTitle: 'Hello, I’m Giovani. How can I help?',
    greetingBody:
      'Describe your task, modality or constraints to receive recommendations from the scientific corpus.',
    questionLabel: 'Your question',
    queryPlaceholder: 'E.g. Which model do you recommend for classifying MRI images?',
    dictate: 'Dictate',
    stopDictation: 'Stop dictation',
    send: 'Send',
    searchingButton: 'Searching…',
    readResponse: 'Read response',
    stop: 'Stop',
    autoRead: 'Automatic playback',
    ready: 'Ready for your question.',
    privacy:
      'AIMER does not store audio recordings. Speech recognition may be processed by your browser service under its own privacy policy.',
    noRecommendation:
      'The corpus does not yet contain enough evidence to make a recommendation.',
    recommendationsOne: '1 recommendation found.',
    recommendationsMany: '{count} recommendations found.',
    model: 'Model {index}',
    confidence: 'Confidence {value}%',
    source: 'Source: {value}. ',
    safety: 'Safety notice.',
    speechStopped: 'Playback stopped.',
    speechInProgress: 'Reading the response…',
    speechFinished: 'Playback finished.',
    speechFailed: 'Voice playback failed. You can read the response on screen.',
    audioCapture: 'No usable microphone was detected.',
    notAllowed: 'Microphone access was denied. Allow it in your browser settings.',
    noSpeech: 'No speech was detected. You can try again.',
    recognitionNetwork: 'The speech-recognition service is unavailable.',
    recognitionAborted: 'Dictation was stopped.',
    recognitionFailed: 'Speech recognition encountered an error.',
    listening: 'Listening… Speak now.',
    dictationFinished: 'Dictation finished. Check the text, then send your question.',
    dictationStartFailed: 'Unable to start dictation. Wait a moment and try again.',
    dictationUnsupported: 'Dictation is not supported by this browser',
    unsupported:
      'Some voice features are not supported by this browser. Text input and on-screen reading remain available.',
    missingQuery: 'Type or dictate a question before sending it.',
    corpusSearching: 'Searching the scientific corpus…',
    sessionExpired: 'Your session has expired. Sign in again to use the assistant.',
    rateLimited: 'Too many requests were sent. Wait one minute and try again.',
    serviceUnavailable: 'The recommendation service is temporarily unavailable.',
    requestFailed: 'The request could not be processed.',
    unexpectedError: 'An unexpected error is preventing the assistant from responding.',
    responseReceived: 'Response received.',
    demoStart: 'Start voice demo',
    demoStop: 'Stop demo',
    demoHelp: 'Experience dictation, RAG analysis and a spoken answer in one flow.',
    demoStepDictation: 'Dictation',
    demoStepAnalysis: 'Analysis',
    demoStepResponse: 'Response',
    demoIntro:
      'Hello, I am Giovani. I will demonstrate my voice capabilities. After this message, ask your question aloud. I will transcribe it, analyse the scientific corpus and read my answer.',
    demoListening: 'Your turn: ask Giovani your question now.',
    demoComplete: 'Demo complete. Giovani listened, analysed and answered.',
    demoStopped: 'Demo stopped.',
    demoUnsupported:
      'This demo requires speech recognition and voice playback in your browser.',
    languageChanged: 'Assistant language set to English.'
  },
  nl: {
    avatarAria: 'Geanimeerde 3D-avatar van Giovani',
    avatarIdle: 'Beschikbaar',
    avatarListening: 'Ik luister',
    avatarThinking: 'Ik zoek',
    avatarSpeaking: 'Ik antwoord',
    avatarSuccess: 'Antwoord voltooid',
    avatarError: 'Er is iets misgegaan',
    avatarBadge: '3D-spraakavatar',
    introSubtitle: 'Doorzoek het corpus met tekst of spraak.',
    introDescription:
      'De avatar luistert, zoekt relevante referenties en beweegt de lippen tijdens het voorlezen van het antwoord.',
    languageLabel: 'Taal van de assistent',
    languageHelp: 'Deze keuze bepaalt de dicteerfunctie, stem en antwoorden.',
    rateLabel: 'Leessnelheid',
    rateHelp: 'Stem en lipbeweging blijven op deze snelheid synchroon.',
    rateChanged: 'Leessnelheid ingesteld op {rate}×.',
    conversationLabel: 'Gesprek met Giovani',
    greetingTitle: 'Hallo, ik ben Giovani. Hoe kan ik u helpen?',
    greetingBody:
      'Beschrijf uw taak, modaliteit of beperkingen om aanbevelingen uit het wetenschappelijke corpus te ontvangen.',
    questionLabel: 'Uw vraag',
    queryPlaceholder: 'Bijv. Welk model raadt u aan voor het classificeren van MRI-beelden?',
    dictate: 'Dicteren',
    stopDictation: 'Dicteren stoppen',
    send: 'Verzenden',
    searchingButton: 'Zoeken…',
    readResponse: 'Antwoord voorlezen',
    stop: 'Stoppen',
    autoRead: 'Automatisch voorlezen',
    ready: 'Klaar voor uw vraag.',
    privacy:
      'AIMER bewaart geen geluidsopnamen. Spraakherkenning kan door de browserdienst worden verwerkt volgens het eigen privacybeleid.',
    noRecommendation:
      'Het corpus bevat nog niet genoeg bewijs om een aanbeveling te doen.',
    recommendationsOne: '1 aanbeveling gevonden.',
    recommendationsMany: '{count} aanbevelingen gevonden.',
    model: 'Model {index}',
    confidence: 'Betrouwbaarheid {value}%',
    source: 'Bron: {value}. ',
    safety: 'Veiligheidsmelding.',
    speechStopped: 'Voorlezen gestopt.',
    speechInProgress: 'Het antwoord wordt voorgelezen…',
    speechFinished: 'Voorlezen voltooid.',
    speechFailed: 'Voorlezen is mislukt. U kunt het antwoord op het scherm lezen.',
    audioCapture: 'Er is geen bruikbare microfoon gedetecteerd.',
    notAllowed: 'Microfoontoegang is geweigerd. Sta deze toe in uw browser.',
    noSpeech: 'Er is geen spraak gedetecteerd. Probeer het opnieuw.',
    recognitionNetwork: 'De spraakherkenningsdienst is niet beschikbaar.',
    recognitionAborted: 'Het dicteren is gestopt.',
    recognitionFailed: 'Er is een fout opgetreden bij de spraakherkenning.',
    listening: 'Ik luister… Spreek nu.',
    dictationFinished: 'Dicteren voltooid. Controleer de tekst en verzend uw vraag.',
    dictationStartFailed: 'Dicteren kan niet worden gestart. Wacht even en probeer opnieuw.',
    dictationUnsupported: 'Dicteren wordt niet ondersteund door deze browser',
    unsupported:
      'Sommige spraakfuncties worden niet ondersteund door deze browser. Tekstinvoer en lezen op het scherm blijven beschikbaar.',
    missingQuery: 'Typ of dicteer een vraag voordat u deze verzendt.',
    corpusSearching: 'Het wetenschappelijke corpus wordt doorzocht…',
    sessionExpired: 'Uw sessie is verlopen. Meld u opnieuw aan om de assistent te gebruiken.',
    rateLimited: 'Er zijn te veel aanvragen verzonden. Wacht een minuut en probeer opnieuw.',
    serviceUnavailable: 'De aanbevelingsdienst is tijdelijk niet beschikbaar.',
    requestFailed: 'De aanvraag kon niet worden verwerkt.',
    unexpectedError: 'Door een onverwachte fout kan de assistent niet antwoorden.',
    responseReceived: 'Antwoord ontvangen.',
    demoStart: 'Spraakdemo starten',
    demoStop: 'Demo stoppen',
    demoHelp: 'Ervaar dicteren, RAG-analyse en een gesproken antwoord in één traject.',
    demoStepDictation: 'Dicteren',
    demoStepAnalysis: 'Analyse',
    demoStepResponse: 'Antwoord',
    demoIntro:
      'Hallo, ik ben Giovani. Ik demonstreer nu mijn spraakfuncties. Stel na dit bericht uw vraag hardop. Ik schrijf ze uit, analyseer het wetenschappelijke corpus en lees mijn antwoord voor.',
    demoListening: 'U bent aan de beurt: stel Giovani nu uw vraag.',
    demoComplete: 'Demo voltooid. Giovani heeft geluisterd, geanalyseerd en geantwoord.',
    demoStopped: 'Demo gestopt.',
    demoUnsupported:
      'Deze demo vereist spraakherkenning en spraakweergave in uw browser.',
    languageChanged: 'De taal van de assistent is ingesteld op Nederlands.'
  },
  de: {
    avatarAria: 'Animierter 3D-Avatar von Giovani',
    avatarIdle: 'Verfügbar',
    avatarListening: 'Ich höre zu',
    avatarThinking: 'Ich suche',
    avatarSpeaking: 'Ich antworte',
    avatarSuccess: 'Antwort abgeschlossen',
    avatarError: 'Ein Fehler ist aufgetreten',
    avatarBadge: '3D-Sprachavatar',
    introSubtitle: 'Fragen Sie den Korpus per Text oder Sprache ab.',
    introDescription:
      'Der Avatar hört zu, sucht relevante Referenzen und bewegt beim Vorlesen der Antwort die Lippen.',
    languageLabel: 'Sprache des Assistenten',
    languageHelp: 'Diese Auswahl steuert Diktat, Stimme und Antworten.',
    rateLabel: 'Lesegeschwindigkeit',
    rateHelp: 'Stimme und Lippenbewegung bleiben bei diesem Tempo synchron.',
    rateChanged: 'Lesegeschwindigkeit auf {rate}× eingestellt.',
    conversationLabel: 'Unterhaltung mit Giovani',
    greetingTitle: 'Hallo, ich bin Giovani. Wie kann ich Ihnen helfen?',
    greetingBody:
      'Beschreiben Sie Ihre Aufgabe, Modalität oder Einschränkungen, um Empfehlungen aus dem wissenschaftlichen Korpus zu erhalten.',
    questionLabel: 'Ihre Frage',
    queryPlaceholder: 'Z. B. Welches Modell empfehlen Sie zur Klassifizierung von MRT-Bildern?',
    dictate: 'Diktieren',
    stopDictation: 'Diktat beenden',
    send: 'Senden',
    searchingButton: 'Suche…',
    readResponse: 'Antwort vorlesen',
    stop: 'Stoppen',
    autoRead: 'Automatisch vorlesen',
    ready: 'Bereit für Ihre Frage.',
    privacy:
      'AIMER speichert keine Audioaufnahmen. Die Spracherkennung kann gemäß den Datenschutzrichtlinien Ihres Browserdienstes verarbeitet werden.',
    noRecommendation:
      'Der Korpus enthält noch nicht genügend Belege für eine Empfehlung.',
    recommendationsOne: '1 Empfehlung gefunden.',
    recommendationsMany: '{count} Empfehlungen gefunden.',
    model: 'Modell {index}',
    confidence: 'Konfidenz {value} %',
    source: 'Quelle: {value}. ',
    safety: 'Sicherheitshinweis.',
    speechStopped: 'Wiedergabe gestoppt.',
    speechInProgress: 'Die Antwort wird vorgelesen…',
    speechFinished: 'Wiedergabe beendet.',
    speechFailed: 'Die Sprachausgabe ist fehlgeschlagen. Sie können die Antwort lesen.',
    audioCapture: 'Es wurde kein verwendbares Mikrofon erkannt.',
    notAllowed: 'Der Mikrofonzugriff wurde verweigert. Erlauben Sie ihn im Browser.',
    noSpeech: 'Es wurde keine Sprache erkannt. Versuchen Sie es erneut.',
    recognitionNetwork: 'Der Spracherkennungsdienst ist nicht verfügbar.',
    recognitionAborted: 'Das Diktat wurde beendet.',
    recognitionFailed: 'Bei der Spracherkennung ist ein Fehler aufgetreten.',
    listening: 'Ich höre zu… Sprechen Sie jetzt.',
    dictationFinished: 'Diktat beendet. Prüfen Sie den Text und senden Sie Ihre Frage.',
    dictationStartFailed: 'Das Diktat kann nicht gestartet werden. Versuchen Sie es erneut.',
    dictationUnsupported: 'Diktat wird von diesem Browser nicht unterstützt',
    unsupported:
      'Einige Sprachfunktionen werden von diesem Browser nicht unterstützt. Texteingabe und Bildschirmdarstellung bleiben verfügbar.',
    missingQuery: 'Geben oder diktieren Sie vor dem Senden eine Frage.',
    corpusSearching: 'Der wissenschaftliche Korpus wird durchsucht…',
    sessionExpired: 'Ihre Sitzung ist abgelaufen. Melden Sie sich erneut an.',
    rateLimited: 'Zu viele Anfragen wurden gesendet. Warten Sie eine Minute.',
    serviceUnavailable: 'Der Empfehlungsdienst ist vorübergehend nicht verfügbar.',
    requestFailed: 'Die Anfrage konnte nicht verarbeitet werden.',
    unexpectedError: 'Ein unerwarteter Fehler verhindert eine Antwort des Assistenten.',
    responseReceived: 'Antwort erhalten.',
    demoStart: 'Sprachdemo starten',
    demoStop: 'Demo beenden',
    demoHelp: 'Erleben Sie Diktat, RAG-Analyse und Sprachantwort in einem Ablauf.',
    demoStepDictation: 'Diktat',
    demoStepAnalysis: 'Analyse',
    demoStepResponse: 'Antwort',
    demoIntro:
      'Hallo, ich bin Giovani. Ich demonstriere jetzt meine Sprachfunktionen. Stellen Sie nach dieser Nachricht Ihre Frage laut. Ich transkribiere sie, analysiere den wissenschaftlichen Korpus und lese meine Antwort vor.',
    demoListening: 'Sie sind dran: Stellen Sie Giovani jetzt Ihre Frage.',
    demoComplete: 'Demo abgeschlossen. Giovani hat zugehört, analysiert und geantwortet.',
    demoStopped: 'Demo beendet.',
    demoUnsupported:
      'Diese Demo benötigt Spracherkennung und Sprachausgabe in Ihrem Browser.',
    languageChanged: 'Die Sprache des Assistenten wurde auf Deutsch eingestellt.'
  }
};

document.addEventListener('DOMContentLoaded', () => {
  const assistant = document.getElementById('voice-assistant');
  if (!assistant) return;

  const form = document.getElementById('voice-assistant-form');
  const queryInput = document.getElementById('voice-assistant-query');
  const conversation = document.getElementById('voice-assistant-conversation');
  const microphoneButton = document.getElementById('voice-assistant-microphone');
  const microphoneIcon = microphoneButton.querySelector('i');
  const microphoneLabel = microphoneButton.querySelector('span');
  const submitButton = document.getElementById('voice-assistant-submit');
  const submitIcon = submitButton.querySelector('i');
  const submitLabel = submitButton.querySelector('span');
  const readButton = document.getElementById('voice-assistant-read');
  const stopButton = document.getElementById('voice-assistant-stop');
  const autoReadInput = document.getElementById('voice-assistant-auto-read');
  const status = document.getElementById('voice-assistant-status');
  const support = document.getElementById('voice-assistant-support');
  const counter = document.getElementById('voice-assistant-counter');
  const avatar = document.getElementById('voice-assistant-avatar');
  const avatarStateLabel = document.getElementById('voice-assistant-avatar-state');
  const languageSelect = document.getElementById('voice-assistant-language');
  const rateSelect = document.getElementById('voice-assistant-rate');
  const demoButton = document.getElementById('voice-assistant-demo');
  const demoIcon = demoButton.querySelector('i');
  const demoLabel = demoButton.querySelector('span');
  const demoProgress = document.getElementById('voice-assistant-demo-progress');
  const demoSteps = Array.from(demoProgress.querySelectorAll('[data-demo-step]'));
  const endpoint = assistant.dataset.endpoint;
  const avatarUrl = assistant.dataset.avatarUrl;
  const supportedLocales = Object.values(VOICE_ASSISTANT_LANGUAGES);
  const supportedSpeechRates = [0.75, 1, 1.25, 1.5];
  let savedLanguage = '';
  let savedSpeechRate = 1;
  try {
    savedLanguage = window.localStorage.getItem('aimer-assistant-language') || '';
    savedSpeechRate = Number.parseFloat(
      window.localStorage.getItem('aimer-assistant-rate') || '1'
    );
  } catch (error) {
    savedLanguage = '';
    savedSpeechRate = 1;
  }
  const browserLanguages = navigator.languages || [navigator.language];
  const browserLanguage = supportedLocales.find(locale =>
    browserLanguages.some(language =>
      language.toLowerCase().startsWith(locale.slice(0, 2).toLowerCase())
    )
  );
  const configuredLanguage = assistant.dataset.language || 'fr-FR';
  let speechLanguage = supportedLocales.includes(savedLanguage)
    ? savedLanguage
    : browserLanguage || configuredLanguage;
  if (!supportedLocales.includes(speechLanguage)) speechLanguage = 'fr-FR';
  let currentLanguage = speechLanguage.slice(0, 2).toLowerCase();
  languageSelect.value = speechLanguage;
  let speechRate = supportedSpeechRates.includes(savedSpeechRate) ? savedSpeechRate : 1;
  rateSelect.value = String(speechRate);
  avatar.dataset.readingRate = String(speechRate);
  avatar.style.setProperty('--speech-rate', String(speechRate));
  assistant.lang = currentLanguage;
  const maxLength = Number.parseInt(assistant.dataset.maxLength, 10) || 2000;
  const lipSyncBaseInterval = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ? 140
    : 72;
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const synthesis = window.speechSynthesis;
  const supportsSpeechOutput =
    Boolean(synthesis) && typeof window.SpeechSynthesisUtterance === 'function';
  const supportsVoiceDemo = Boolean(Recognition) && supportsSpeechOutput;

  let recognition = null;
  let recognitionFailed = false;
  let isListening = false;
  let isLoading = false;
  let isSpeaking = false;
  let dictatedBase = '';
  let lastResponseSpeech = '';
  let activeUtterance = null;
  let lipSyncTimer = null;
  let lipSyncText = '';
  let lipSyncCursor = 0;
  let feedbackState = null;
  let feedbackTimer = null;
  let demoMode = false;
  let demoAwaitingDictation = false;
  let demoPreviousAutoRead = false;

  const translate = (key, replacements = {}) => {
    const translations =
      VOICE_ASSISTANT_TRANSLATIONS[currentLanguage] || VOICE_ASSISTANT_TRANSLATIONS.fr;
    let value = translations[key] || VOICE_ASSISTANT_TRANSLATIONS.fr[key] || key;
    Object.entries(replacements).forEach(([name, replacement]) => {
      value = value.split(`{${name}}`).join(String(replacement));
    });
    return value;
  };

  const applyTranslations = () => {
    assistant.querySelectorAll('[data-i18n]').forEach(element => {
      element.textContent = translate(element.dataset.i18n);
    });
    assistant.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
      element.setAttribute('placeholder', translate(element.dataset.i18nPlaceholder));
    });
    assistant.querySelectorAll('[data-i18n-aria-label]').forEach(element => {
      element.setAttribute('aria-label', translate(element.dataset.i18nAriaLabel));
    });
    avatar.setAttribute('aria-label', translate('avatarAria'));
    assistant.lang = currentLanguage;
  };

  const setDemoPhase = phase => {
    const phases = ['dictation', 'analysis', 'response'];
    const phaseIndex = phase === 'complete' ? phases.length : phases.indexOf(phase);
    const visible = phaseIndex >= 0;
    demoProgress.hidden = !visible;
    demoProgress.setAttribute('aria-hidden', String(!visible));
    demoSteps.forEach((step, index) => {
      step.classList.toggle('is-active', phase !== 'complete' && index === phaseIndex);
      step.classList.toggle('is-complete', phase === 'complete' || index < phaseIndex);
    });
  };

  const setDemoButtonState = active => {
    demoButton.classList.toggle('btn-label-danger', active);
    demoButton.classList.toggle('btn-label-primary', !active);
    demoButton.setAttribute('aria-pressed', String(active));
    demoIcon.className = active
      ? 'icon-base bx bx-stop-circle me-1'
      : 'icon-base bx bx-play-circle me-1';
    demoLabel.textContent = translate(active ? 'demoStop' : 'demoStart');
  };

  const finishDemo = ({statusKey = null, complete = false} = {}) => {
    demoMode = false;
    demoAwaitingDictation = false;
    autoReadInput.checked = demoPreviousAutoRead;
    setDemoButtonState(false);
    setDemoPhase(complete ? 'complete' : null);
    demoButton.disabled = !supportsVoiceDemo || isLoading;
    microphoneButton.disabled = isLoading || !Recognition;
    submitButton.disabled = isLoading || isListening;
    languageSelect.disabled = isLoading || isListening;
    rateSelect.disabled = !supportsSpeechOutput || isLoading || isListening || isSpeaking;
    if (complete) showFeedback('success');
    if (statusKey) setStatus(translate(statusKey));
  };

  const refreshAvatarState = () => {
    const state = isSpeaking
      ? 'speaking'
      : isListening
        ? 'listening'
        : isLoading
          ? 'thinking'
          : feedbackState || 'idle';
    const labelKeys = {
      idle: 'avatarIdle',
      listening: 'avatarListening',
      thinking: 'avatarThinking',
      speaking: 'avatarSpeaking',
      success: 'avatarSuccess',
      error: 'avatarError'
    };
    avatar.dataset.state = state;
    avatarStateLabel.textContent = translate(labelKeys[state]);
  };

  const clearFeedback = (refresh = true) => {
    if (feedbackTimer) window.clearTimeout(feedbackTimer);
    feedbackTimer = null;
    feedbackState = null;
    if (refresh) refreshAvatarState();
  };

  const showFeedback = (state, duration = 1800) => {
    clearFeedback(false);
    feedbackState = state;
    refreshAvatarState();
    feedbackTimer = window.setTimeout(() => {
      feedbackTimer = null;
      feedbackState = null;
      refreshAvatarState();
    }, duration);
  };

  const visemeForPosition = (text, index) => {
    const normalized = text
      .slice(index, index + 3)
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .toLowerCase();
    const firstCharacter = normalized.slice(0, 1);
    if (!firstCharacter || /[\s.,;:!?'-]/.test(firstCharacter)) return 'rest';
    if (/^[bmp]/.test(normalized)) return 'closed';
    if (/^(ou|oo|oe|au|eu|[ouqw])/.test(normalized)) return 'round';
    if (/^(a|aa|ah)/.test(normalized)) return 'open';
    if (/^(ij|ie|ei|ey|[eiy])/.test(normalized)) return 'wide';
    return index % 2 === 0 ? 'open' : 'closed';
  };

  const updateViseme = (text, index) => {
    avatar.dataset.viseme = visemeForPosition(text, index);
    avatar.dataset.lipIndex = String(index);
  };

  const lipFrameDelay = character => {
    const pauseWeight = /[.!?;:]/.test(character)
      ? 3.5
      : /[,\s]/.test(character)
        ? 1.4
        : 1;
    return Math.max(32, Math.round((lipSyncBaseInterval * pauseWeight) / speechRate));
  };

  const scheduleLipFrame = () => {
    if (!lipSyncText || !isSpeaking) return;
    if (lipSyncCursor >= lipSyncText.length) {
      avatar.dataset.viseme = 'rest';
      return;
    }

    const character = lipSyncText[lipSyncCursor];
    updateViseme(lipSyncText, lipSyncCursor);
    lipSyncCursor += 1;
    lipSyncTimer = window.setTimeout(scheduleLipFrame, lipFrameDelay(character));
  };

  const stopLipSync = () => {
    if (lipSyncTimer) window.clearTimeout(lipSyncTimer);
    lipSyncTimer = null;
    lipSyncText = '';
    lipSyncCursor = 0;
    avatar.dataset.viseme = 'rest';
    avatar.dataset.lipIndex = '0';
    isSpeaking = false;
    rateSelect.disabled = !supportsSpeechOutput || isLoading || isListening || demoMode;
    refreshAvatarState();
  };

  const startLipSync = text => {
    if (lipSyncTimer) window.clearTimeout(lipSyncTimer);
    clearFeedback(false);
    lipSyncText = text;
    lipSyncCursor = 0;
    isSpeaking = true;
    rateSelect.disabled = true;
    avatar.dataset.readingRate = String(speechRate);
    refreshAvatarState();
    scheduleLipFrame();
  };

  const syncLipToBoundary = (event, text) => {
    const requestedIndex = Number.isFinite(event.charIndex) ? event.charIndex : 0;
    const boundaryIndex = Math.max(0, Math.min(requestedIndex, text.length - 1));
    lipSyncCursor = boundaryIndex;
    updateViseme(text, boundaryIndex);
    lipSyncCursor = boundaryIndex + 1;
  };

  const setStatus = message => {
    status.textContent = message;
  };

  const updateCounter = () => {
    counter.textContent = `${queryInput.value.length} / ${maxLength}`;
  };

  const scrollConversation = () => {
    conversation.scrollTop = conversation.scrollHeight;
  };

  const createElement = (tagName, className, text) => {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };

  const appendMessage = (kind, content) => {
    const message = createElement(
      'div',
      `voice-assistant__message voice-assistant__message--${kind}`
    );

    if (kind === 'assistant') {
      const avatar = createElement('span', 'voice-assistant__message-avatar');
      avatar.setAttribute('aria-hidden', 'true');
      const avatarImage = createElement('img', 'voice-assistant__message-avatar-image');
      avatarImage.src = avatarUrl;
      avatarImage.alt = '';
      avatar.append(avatarImage);
      message.append(avatar);
    }

    const bubble = createElement('div', 'voice-assistant__bubble');
    if (typeof content === 'string') {
      bubble.textContent = content;
    } else {
      bubble.append(content);
    }
    message.append(bubble);
    conversation.append(message);
    scrollConversation();
  };

  const normalizedConfidence = value => {
    const confidence = Number(value);
    if (!Number.isFinite(confidence)) return null;
    return Math.round(Math.min(1, Math.max(0, confidence)) * 100);
  };

  const buildResponse = payload => {
    const wrapper = document.createDocumentFragment();
    const recommendations = Array.isArray(payload.recommended_models)
      ? payload.recommended_models
      : [];
    const speechParts = [];

    if (recommendations.length === 0) {
      const reason =
        payload.no_recommendation_reason ||
        translate('noRecommendation');
      wrapper.append(createElement('p', 'mb-0', reason));
      speechParts.push(reason);
    } else {
      const introduction = translate(
        recommendations.length === 1 ? 'recommendationsOne' : 'recommendationsMany',
        {count: recommendations.length}
      );
      wrapper.append(createElement('p', 'fw-medium mb-3', introduction));
      speechParts.push(introduction);

      recommendations.forEach((recommendation, index) => {
        const item = createElement('article', 'voice-assistant__recommendation');
        const heading = createElement('div', 'd-flex flex-wrap align-items-center gap-2 mb-1');
        const modelName =
          recommendation.model_name || translate('model', {index: index + 1});
        heading.append(createElement('h4', 'h6 mb-0', modelName));

        const confidence = normalizedConfidence(recommendation.confidence);
        if (confidence !== null) {
          heading.append(
            createElement(
              'span',
              'badge bg-label-primary',
              translate('confidence', {value: confidence})
            )
          );
        }
        item.append(heading);

        if (recommendation.rationale) {
          item.append(createElement('p', 'mb-1', recommendation.rationale));
        }

        const evidence = Array.isArray(recommendation.evidence)
          ? recommendation.evidence[0]
          : null;
        if (evidence && evidence.snippet) {
          const source = createElement('div', 'voice-assistant__source small text-muted');
          const sourceName = evidence.source
            ? translate('source', {value: evidence.source})
            : '';
          source.textContent = `${sourceName}${evidence.snippet}`;
          item.append(source);
        }

        wrapper.append(item);
        speechParts.push(
          `${index + 1}. ${modelName}. ${recommendation.rationale || ''}`.trim()
        );
      });
    }

    if (payload.safety_notice) {
      const notice = createElement('p', 'alert alert-warning small mb-0 mt-3');
      notice.textContent = payload.safety_notice;
      wrapper.append(notice);
      speechParts.push(`${translate('safety')} ${payload.safety_notice}`);
    }

    return {content: wrapper, speech: speechParts.join(' ')};
  };

  const setLoading = loading => {
    if (loading) clearFeedback(false);
    isLoading = loading;
    submitButton.disabled = loading;
    microphoneButton.disabled = loading || !Recognition;
    languageSelect.disabled = loading || isListening || demoMode;
    rateSelect.disabled =
      !supportsSpeechOutput || loading || isListening || isSpeaking || demoMode;
    demoButton.disabled = !supportsVoiceDemo || (loading && !demoMode);
    queryInput.readOnly = loading;
    submitIcon.className = loading
      ? 'icon-base bx bx-loader-circle voice-assistant__loading-icon me-1'
      : 'icon-base bx bx-send me-1';
    submitLabel.textContent = translate(loading ? 'searchingButton' : 'send');
    refreshAvatarState();
  };

  const stopSpeaking = () => {
    if (!supportsSpeechOutput) return;
    activeUtterance = null;
    synthesis.cancel();
    stopLipSync();
    stopButton.disabled = true;
    setStatus(translate('speechStopped'));
  };

  const speak = (text, {onEnd = null, onError = null} = {}) => {
    if (!supportsSpeechOutput || !text) return;

    if (recognition && isListening) recognition.stop();
    activeUtterance = null;
    synthesis.cancel();
    stopLipSync();
    const utterance = new window.SpeechSynthesisUtterance(text);
    activeUtterance = utterance;
    utterance.lang = speechLanguage;
    utterance.rate = speechRate;

    const availableVoices = synthesis.getVoices();
    const exactLanguage = speechLanguage.toLowerCase();
    const baseLanguage = speechLanguage.slice(0, 2).toLowerCase();
    const matchingVoice =
      availableVoices.find(voice => voice.lang.toLowerCase() === exactLanguage) ||
      availableVoices.find(voice => voice.lang.toLowerCase().startsWith(baseLanguage));
    if (matchingVoice) utterance.voice = matchingVoice;

    utterance.addEventListener('start', () => {
      if (activeUtterance !== utterance) return;
      stopButton.disabled = false;
      startLipSync(text);
      setStatus(translate('speechInProgress'));
    });
    utterance.addEventListener('boundary', event => {
      if (activeUtterance !== utterance) return;
      syncLipToBoundary(event, text);
    });
    utterance.addEventListener('end', () => {
      if (activeUtterance !== utterance) return;
      activeUtterance = null;
      stopButton.disabled = true;
      stopLipSync();
      setStatus(translate('speechFinished'));
      showFeedback('success');
      if (typeof onEnd === 'function') onEnd();
    });
    utterance.addEventListener('error', event => {
      if (activeUtterance !== utterance) return;
      activeUtterance = null;
      stopButton.disabled = true;
      stopLipSync();
      if (event.error !== 'canceled' && event.error !== 'interrupted') {
        setStatus(translate('speechFailed'));
        showFeedback('error');
        if (typeof onError === 'function') onError(event);
      }
    });
    synthesis.speak(utterance);
  };

  const recognitionErrorMessage = error => {
    const messages = {
      'audio-capture': 'audioCapture',
      'not-allowed': 'notAllowed',
      'no-speech': 'noSpeech',
      network: 'recognitionNetwork',
      aborted: 'recognitionAborted'
    };
    return translate(messages[error] || 'recognitionFailed');
  };

  const setListening = listening => {
    if (listening) clearFeedback(false);
    isListening = listening;
    microphoneButton.classList.toggle('is-listening', listening);
    microphoneButton.setAttribute('aria-pressed', String(listening));
    microphoneIcon.className = listening
      ? 'icon-base bx bx-stop-circle me-1'
      : 'icon-base bx bx-microphone me-1';
    microphoneLabel.textContent = translate(listening ? 'stopDictation' : 'dictate');
    submitButton.disabled = listening || isLoading;
    languageSelect.disabled = listening || isLoading || demoMode;
    rateSelect.disabled =
      !supportsSpeechOutput || listening || isLoading || isSpeaking || demoMode;
    refreshAvatarState();
  };

  if (Recognition) {
    recognition = new Recognition();
    recognition.lang = speechLanguage;
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.addEventListener('start', () => {
      recognitionFailed = false;
      setListening(true);
      setStatus(translate('listening'));
    });

    recognition.addEventListener('result', event => {
      let transcript = '';
      for (let index = 0; index < event.results.length; index += 1) {
        transcript += event.results[index][0].transcript;
      }
      queryInput.value = [dictatedBase, transcript.trim()]
        .filter(Boolean)
        .join(' ')
        .slice(0, maxLength);
      updateCounter();
    });

    recognition.addEventListener('error', event => {
      recognitionFailed = true;
      setStatus(recognitionErrorMessage(event.error));
      showFeedback('error');
    });

    recognition.addEventListener('end', () => {
      setListening(false);
      dictatedBase = queryInput.value.trim();
      if (demoMode && demoAwaitingDictation) {
        demoAwaitingDictation = false;
        if (dictatedBase && !recognitionFailed) {
          setDemoPhase('analysis');
          setStatus(translate('corpusSearching'));
          window.setTimeout(() => form.requestSubmit(), 250);
        } else {
          finishDemo();
        }
      } else if (!recognitionFailed) {
        setStatus(
          dictatedBase
            ? translate('dictationFinished')
            : translate('noSpeech')
        );
      }
      if (dictatedBase) queryInput.focus();
    });

    microphoneButton.addEventListener('click', () => {
      if (isListening) {
        recognition.stop();
        return;
      }

      if (isSpeaking) stopSpeaking();
      dictatedBase = queryInput.value.trim();
      recognitionFailed = false;
      try {
        recognition.start();
      } catch (error) {
        setStatus(translate('dictationStartFailed'));
        showFeedback('error');
      }
    });
  }

  const unsupportedFeatures = [];
  if (!Recognition) unsupportedFeatures.push('dictation');
  if (!supportsSpeechOutput) unsupportedFeatures.push('speech');

  const updateUnsupportedMessage = () => {
    if (!Recognition) {
      microphoneButton.disabled = true;
      microphoneButton.title = translate('dictationUnsupported');
    }
    if (!supportsSpeechOutput) {
      readButton.disabled = true;
      stopButton.disabled = true;
      autoReadInput.disabled = true;
      rateSelect.disabled = true;
    }
    demoButton.disabled = !supportsVoiceDemo;
    demoButton.title = supportsVoiceDemo ? '' : translate('demoUnsupported');
    support.hidden = unsupportedFeatures.length === 0;
    support.textContent = unsupportedFeatures.length > 0 ? translate('unsupported') : '';
  };

  queryInput.addEventListener('input', updateCounter);
  readButton.addEventListener('click', () => speak(lastResponseSpeech));
  stopButton.addEventListener('click', stopSpeaking);
  demoButton.addEventListener('click', () => {
    if (demoMode) {
      if (recognition && isListening) {
        recognitionFailed = true;
        recognition.abort();
      }
      if (isSpeaking) stopSpeaking();
      finishDemo({statusKey: 'demoStopped'});
      return;
    }
    if (!supportsVoiceDemo) {
      setStatus(translate('demoUnsupported'));
      return;
    }

    demoMode = true;
    demoAwaitingDictation = false;
    demoPreviousAutoRead = autoReadInput.checked;
    autoReadInput.checked = true;
    queryInput.value = '';
    dictatedBase = '';
    updateCounter();
    setDemoButtonState(true);
    setDemoPhase('dictation');
    languageSelect.disabled = true;
    rateSelect.disabled = true;
    microphoneButton.disabled = true;
    submitButton.disabled = true;
    setStatus(translate('demoIntro'));
    speak(translate('demoIntro'), {
      onEnd: () => {
        if (!demoMode || !recognition) return;
        demoAwaitingDictation = true;
        microphoneButton.disabled = false;
        setStatus(translate('demoListening'));
        try {
          recognition.start();
        } catch (error) {
          setStatus(translate('dictationStartFailed'));
          showFeedback('error');
          finishDemo();
        }
      },
      onError: () => finishDemo()
    });
  });
  rateSelect.addEventListener('change', () => {
    const requestedRate = Number.parseFloat(rateSelect.value);
    speechRate = supportedSpeechRates.includes(requestedRate) ? requestedRate : 1;
    rateSelect.value = String(speechRate);
    avatar.dataset.readingRate = String(speechRate);
    avatar.style.setProperty('--speech-rate', String(speechRate));
    try {
      window.localStorage.setItem('aimer-assistant-rate', String(speechRate));
    } catch (error) {
      // The selected speed still applies for this page when storage is blocked.
    }
    setStatus(translate('rateChanged', {rate: speechRate}));
  });
  languageSelect.addEventListener('change', () => {
    if (recognition && isListening) {
      recognition.abort();
      setListening(false);
    }
    if (isSpeaking) stopSpeaking();

    speechLanguage = supportedLocales.includes(languageSelect.value)
      ? languageSelect.value
      : 'fr-FR';
    currentLanguage = speechLanguage.slice(0, 2).toLowerCase();
    assistant.dataset.language = speechLanguage;
    if (recognition) recognition.lang = speechLanguage;
    try {
      window.localStorage.setItem('aimer-assistant-language', speechLanguage);
    } catch (error) {
      // The selected language still applies for this page when storage is blocked.
    }

    lastResponseSpeech = '';
    readButton.disabled = true;
    applyTranslations();
    setLoading(isLoading);
    refreshAvatarState();
    updateUnsupportedMessage();
    setStatus(translate('languageChanged'));
  });

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const query = queryInput.value.trim();
    if (!query || isLoading) {
      if (!query) {
        setStatus(translate('missingQuery'));
        queryInput.focus();
      }
      return;
    }

    if (isListening) recognition.stop();
    if (demoMode) setDemoPhase('analysis');
    appendMessage('user', query);
    setLoading(true);
    setStatus(translate('corpusSearching'));

    try {
      const parameters = new URLSearchParams({
        q: query,
        top_k: '3',
        language: currentLanguage
      });
      const response = await fetch(`${endpoint}?${parameters.toString()}`, {
        credentials: 'same-origin',
        headers: {'Accept': 'application/json'}
      });
      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        const errorMessages = {
          401: translate('sessionExpired'),
          429: translate('rateLimited'),
          503: translate('serviceUnavailable')
        };
        throw new Error(
          errorMessages[response.status] ||
            payload.error ||
            translate('requestFailed')
        );
      }

      const rendered = buildResponse(payload);
      appendMessage('assistant', rendered.content);
      lastResponseSpeech = rendered.speech;
      readButton.disabled = !supportsSpeechOutput || !lastResponseSpeech;
      queryInput.value = '';
      dictatedBase = '';
      updateCounter();
      setStatus(translate('responseReceived'));
      if (demoMode) setDemoPhase('response');
      if (autoReadInput.checked) {
        speak(lastResponseSpeech, {
          onEnd: demoMode
            ? () => finishDemo({statusKey: 'demoComplete', complete: true})
            : null,
          onError: demoMode ? () => finishDemo() : null
        });
      } else if (demoMode) {
        finishDemo({statusKey: 'demoComplete', complete: true});
      } else {
        showFeedback('success');
      }
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : translate('unexpectedError');
      appendMessage('assistant', message);
      setStatus(message);
      showFeedback('error');
      if (demoMode) finishDemo();
    } finally {
      setLoading(false);
      queryInput.focus();
    }
  });

  window.addEventListener('pagehide', () => {
    if (recognition && isListening) recognition.abort();
    if (supportsSpeechOutput) {
      activeUtterance = null;
      synthesis.cancel();
      stopLipSync();
    }
  });

  applyTranslations();
  setDemoButtonState(false);
  setDemoPhase(null);
  refreshAvatarState();
  updateUnsupportedMessage();
  updateCounter();
});
