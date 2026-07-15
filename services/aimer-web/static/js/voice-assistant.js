/**
 * Browser-native speech input and output for the authenticated RAG assistant.
 */
'use strict';

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
  const endpoint = assistant.dataset.endpoint;
  const speechLanguage = assistant.dataset.language || 'fr-FR';
  const maxLength = Number.parseInt(assistant.dataset.maxLength, 10) || 2000;
  const lipSyncInterval = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ? 140
    : 75;
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const synthesis = window.speechSynthesis;
  const supportsSpeechOutput =
    Boolean(synthesis) && typeof window.SpeechSynthesisUtterance === 'function';

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

  const refreshAvatarState = () => {
    const state = isSpeaking
      ? 'speaking'
      : isListening
        ? 'listening'
        : isLoading
          ? 'thinking'
          : 'idle';
    const labels = {
      idle: 'Disponible',
      listening: 'Je vous écoute',
      thinking: 'Je recherche',
      speaking: 'Je vous réponds'
    };
    avatar.dataset.state = state;
    avatarStateLabel.textContent = labels[state];
  };

  const visemeForCharacter = character => {
    const normalized = character
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .toLowerCase();
    if (!normalized || /[\s.,;:!?'-]/.test(normalized)) return 'rest';
    if (/[bmp]/.test(normalized)) return 'closed';
    if (/[ouqw]/.test(normalized)) return 'round';
    if (/[a]/.test(normalized)) return 'open';
    if (/[eiy]/.test(normalized)) return 'wide';
    return lipSyncCursor % 2 === 0 ? 'open' : 'closed';
  };

  const updateViseme = character => {
    avatar.dataset.viseme = visemeForCharacter(character);
  };

  const stopLipSync = () => {
    if (lipSyncTimer) window.clearInterval(lipSyncTimer);
    lipSyncTimer = null;
    lipSyncText = '';
    lipSyncCursor = 0;
    avatar.dataset.viseme = 'rest';
    isSpeaking = false;
    refreshAvatarState();
  };

  const startLipSync = text => {
    if (lipSyncTimer) window.clearInterval(lipSyncTimer);
    lipSyncText = text;
    lipSyncCursor = 0;
    isSpeaking = true;
    refreshAvatarState();
    lipSyncTimer = window.setInterval(() => {
      if (!lipSyncText) return;
      updateViseme(lipSyncText[lipSyncCursor % lipSyncText.length]);
      lipSyncCursor += 1;
    }, lipSyncInterval);
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
      avatar.append(createElement('i', 'icon-base bx bx-bot'));
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
        'Le corpus ne contient pas encore assez d’éléments pour formuler une recommandation.';
      wrapper.append(createElement('p', 'mb-0', reason));
      speechParts.push(reason);
    } else {
      const introduction = `${recommendations.length} recommandation${
        recommendations.length > 1 ? 's' : ''
      } trouvée${recommendations.length > 1 ? 's' : ''}.`;
      wrapper.append(createElement('p', 'fw-medium mb-3', introduction));
      speechParts.push(introduction);

      recommendations.forEach((recommendation, index) => {
        const item = createElement('article', 'voice-assistant__recommendation');
        const heading = createElement('div', 'd-flex flex-wrap align-items-center gap-2 mb-1');
        const modelName = recommendation.model_name || `Modèle ${index + 1}`;
        heading.append(createElement('h4', 'h6 mb-0', modelName));

        const confidence = normalizedConfidence(recommendation.confidence);
        if (confidence !== null) {
          heading.append(
            createElement('span', 'badge bg-label-primary', `Confiance ${confidence} %`)
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
          const sourceName = evidence.source ? `Source : ${evidence.source}. ` : '';
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
      speechParts.push(`Note de sécurité. ${payload.safety_notice}`);
    }

    return {content: wrapper, speech: speechParts.join(' ')};
  };

  const setLoading = loading => {
    isLoading = loading;
    submitButton.disabled = loading;
    microphoneButton.disabled = loading || !Recognition;
    queryInput.readOnly = loading;
    submitIcon.className = loading
      ? 'icon-base bx bx-loader-circle voice-assistant__loading-icon me-1'
      : 'icon-base bx bx-send me-1';
    submitLabel.textContent = loading ? 'Recherche…' : 'Envoyer';
    refreshAvatarState();
  };

  const stopSpeaking = () => {
    if (!supportsSpeechOutput) return;
    activeUtterance = null;
    synthesis.cancel();
    stopLipSync();
    stopButton.disabled = true;
    setStatus('Lecture arrêtée.');
  };

  const speak = text => {
    if (!supportsSpeechOutput || !text) return;

    if (recognition && isListening) recognition.stop();
    activeUtterance = null;
    synthesis.cancel();
    stopLipSync();
    const utterance = new window.SpeechSynthesisUtterance(text);
    activeUtterance = utterance;
    utterance.lang = speechLanguage;
    utterance.rate = 1;

    const matchingVoice = synthesis
      .getVoices()
      .find(voice => voice.lang.toLowerCase().startsWith(speechLanguage.slice(0, 2).toLowerCase()));
    if (matchingVoice) utterance.voice = matchingVoice;

    utterance.addEventListener('start', () => {
      if (activeUtterance !== utterance) return;
      stopButton.disabled = false;
      startLipSync(text);
      setStatus('Lecture de la réponse en cours…');
    });
    utterance.addEventListener('boundary', event => {
      if (activeUtterance !== utterance) return;
      lipSyncCursor = Math.max(0, Math.min(event.charIndex, text.length - 1));
      updateViseme(text.slice(lipSyncCursor, lipSyncCursor + 1));
    });
    utterance.addEventListener('end', () => {
      if (activeUtterance !== utterance) return;
      activeUtterance = null;
      stopButton.disabled = true;
      stopLipSync();
      setStatus('Lecture terminée.');
    });
    utterance.addEventListener('error', event => {
      if (activeUtterance !== utterance) return;
      activeUtterance = null;
      stopButton.disabled = true;
      stopLipSync();
      if (event.error !== 'canceled' && event.error !== 'interrupted') {
        setStatus('La lecture vocale a échoué. Vous pouvez lire la réponse à l’écran.');
      }
    });
    synthesis.speak(utterance);
  };

  const recognitionErrorMessage = error => {
    const messages = {
      'audio-capture': 'Aucun microphone utilisable n’a été détecté.',
      'not-allowed': 'L’accès au microphone a été refusé. Autorisez-le dans le navigateur.',
      'no-speech': 'Aucune parole n’a été détectée. Vous pouvez réessayer.',
      network: 'Le service de reconnaissance vocale est indisponible.',
      aborted: 'La dictée a été arrêtée.'
    };
    return messages[error] || 'La dictée vocale a rencontré une erreur.';
  };

  const setListening = listening => {
    isListening = listening;
    microphoneButton.classList.toggle('is-listening', listening);
    microphoneButton.setAttribute('aria-pressed', String(listening));
    microphoneIcon.className = listening
      ? 'icon-base bx bx-stop-circle me-1'
      : 'icon-base bx bx-microphone me-1';
    microphoneLabel.textContent = listening ? 'Arrêter la dictée' : 'Dicter';
    submitButton.disabled = listening || isLoading;
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
      setStatus('Écoute en cours… Parlez maintenant.');
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
    });

    recognition.addEventListener('end', () => {
      setListening(false);
      dictatedBase = queryInput.value.trim();
      if (!recognitionFailed) {
        setStatus(
          dictatedBase
            ? 'Dictée terminée. Vérifiez le texte puis envoyez votre question.'
            : 'Aucune parole n’a été détectée. Vous pouvez réessayer.'
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
        setStatus('Impossible de démarrer la dictée. Attendez un instant puis réessayez.');
      }
    });
  }

  const unsupportedFeatures = [];
  if (!Recognition) {
    microphoneButton.disabled = true;
    microphoneButton.title = 'Dictée non prise en charge par ce navigateur';
    unsupportedFeatures.push('la dictée');
  }
  if (!supportsSpeechOutput) {
    readButton.disabled = true;
    stopButton.disabled = true;
    autoReadInput.disabled = true;
    unsupportedFeatures.push('la lecture vocale');
  }
  if (unsupportedFeatures.length > 0) {
    support.hidden = false;
    support.textContent = `Ce navigateur ne prend pas en charge ${unsupportedFeatures.join(
      ' et '
    )}. La saisie et la lecture à l’écran restent disponibles.`;
  }

  queryInput.addEventListener('input', updateCounter);
  readButton.addEventListener('click', () => speak(lastResponseSpeech));
  stopButton.addEventListener('click', stopSpeaking);

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const query = queryInput.value.trim();
    if (!query || isLoading) {
      if (!query) {
        setStatus('Saisissez ou dictez une question avant de l’envoyer.');
        queryInput.focus();
      }
      return;
    }

    if (isListening) recognition.stop();
    appendMessage('user', query);
    setLoading(true);
    setStatus('Recherche dans le corpus scientifique…');

    try {
      const parameters = new URLSearchParams({q: query, top_k: '3'});
      const response = await fetch(`${endpoint}?${parameters.toString()}`, {
        credentials: 'same-origin',
        headers: {'Accept': 'application/json'}
      });
      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        const errorMessages = {
          401: 'Votre session a expiré. Reconnectez-vous pour utiliser l’assistant.',
          429: 'Trop de demandes ont été envoyées. Patientez une minute puis réessayez.',
          503: 'Le service de recommandation est momentanément indisponible.'
        };
        throw new Error(
          errorMessages[response.status] ||
            payload.error ||
            'La demande n’a pas pu être traitée.'
        );
      }

      const rendered = buildResponse(payload);
      appendMessage('assistant', rendered.content);
      lastResponseSpeech = rendered.speech;
      readButton.disabled = !supportsSpeechOutput || !lastResponseSpeech;
      queryInput.value = '';
      dictatedBase = '';
      updateCounter();
      setStatus('Réponse reçue.');
      if (autoReadInput.checked) speak(lastResponseSpeech);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'Une erreur inattendue empêche l’assistant de répondre.';
      appendMessage('assistant', message);
      setStatus(message);
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

  refreshAvatarState();
  updateCounter();
});
