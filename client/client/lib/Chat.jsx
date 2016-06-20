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

    submit(userInputRelayer, e) {
        if (e.which === 13) {
            const value = e.target.value;
            userInputRelayer(value);
        }
    }

    render() {
        const { waitingForBot, userInputRelayer } = this.props;
        return (
            <input
                  className="chat-input"
                  onKeyPress={ waitingForBot ? noop : this.submit.bind(this, userInputRelayer) }
                  placeholder="ask a question"/>
         );
    }
}
ChatInput.propTypes = propTypes;
module.exports = ChatInput;
