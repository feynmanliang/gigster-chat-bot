import React from 'react';

const noop = () => {};

const { PropTypes } = React;
const { func, bool } = PropTypes;

const propTypes = {
    userInputRelayer: func.isRequired,
    waitingForBot: bool.isRequired
};

class ChatInput extends React.Component {
    constructor() {
        super();
    }

    handleKeyPress(e) {
      if (e.which === 13) { // pressed enter
        const { waitingForBot, userInputRelayer } = this.props;
        if (!waitingForBot) {
          userInputRelayer(e.target.value);
          this.refs.chatInput.value = "";
        }
      }
    }

    render() {
        return (
            <input
                  className="chat-input"
                  ref='chatInput'
                  onKeyPress={ this.handleKeyPress.bind(this) }
                  placeholder="ask a question"/>
         );
    }
}
ChatInput.propTypes = propTypes;
module.exports = ChatInput;
