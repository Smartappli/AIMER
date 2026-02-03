/**
 *  Pages Authentication
 * This Used in all authentication pages
 */

'use strict';
const formAuthentication = document.querySelector('#formAuthentication');
const btnSubmit = document.getElementById('btnSubmit');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');

document.addEventListener('DOMContentLoaded', function (e) {
  (function () {
    // Validate the form using FormValidation library
    if (formAuthentication) {
      const fv = FormValidation.formValidation(formAuthentication, {
        fields: {
          username: {
            validators: {
              notEmpty: {
                message: 'Please enter username'
              },
              stringLength: {
                min: 4,
                message: 'Username must be more than 4 characters'
              }
            }
          },
          email: {
            validators: {
              notEmpty: {
                message: 'Please enter your email'
              },
              emailAddress: {
                message: 'Please enter a valid email address'
              }
            }
          },
          'email-username': {
            validators: {
              notEmpty: {
                message: 'Please enter email / username'
              },
              stringLength: {
                min: 4,
                message: 'Username must be more than 4 characters'
              }
            }
          },
          password: {
            validators: {
              notEmpty: {
                message: 'Please enter your password'
              },
              stringLength: {
                min: 4,
                message: 'Password must be more than 4 characters'
              }
            }
          },
          'confirm-password': {
            validators: {
              notEmpty: {
                message: 'Please confirm password'
              },
              identical: {
                compare: () => formAuthentication.querySelector('[name="password"]').value,
                message: 'The password and its confirmation do not match'
              },
              stringLength: {
                min: 4,
                message: 'Password must be more than 4 characters'
              }
            }
          },
          terms: {
            validators: {
              notEmpty: {
                message: 'Please agree to terms & conditions'
              }
            }
          }
        },
        plugins: {
          trigger: new FormValidation.plugins.Trigger(),
          bootstrap5: new FormValidation.plugins.Bootstrap5({
            eleValidClass: '',
            rowSelector: '.form-control-validation'
          }),
          submitButton: new FormValidation.plugins.SubmitButton(),

          defaultSubmit: new FormValidation.plugins.DefaultSubmit(),
          autoFocus: new FormValidation.plugins.AutoFocus()
        },
        init: instance => {
          instance.on('plugins.message.placed', e => {
            if (e.element.parentElement.classList.contains('input-group')) {
              e.element.parentElement.insertAdjacentElement('afterend', e.messageElement);
            }
          });
        }
      });

      if (btnSubmit && btnText && btnLoader) {
        // Show loading state on form submission
        btnSubmit.addEventListener('click', function (event) {
          event.preventDefault(); // Prevent default form submission

          // Check if the form is valid
          fv.validate().then(function (status) {
            if (status === 'Valid') {
              // If the form is valid, show loading state
              btnSubmit.classList.add('disabled');
              btnText.textContent = 'Sending email... ';
              btnLoader.classList.remove('visually-hidden');
            }
          });
        });
      }
    }
  })();
});
