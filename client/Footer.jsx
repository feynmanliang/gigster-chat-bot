import React from 'react';

export default Footer = React.createClass({
  propTypes: {
    channel: React.PropTypes.string.isRequired,
    newMessage: React.PropTypes.func.isRequired
  },
  handleKeyPress(event) {
    const textInput = React.findDOMNode(this.refs.textInput)
    const messageText = textInput.value;
    if (!!textInput && event.charCode == 13) { // pressed Return
      event.preventDefault();
      this.props.newMessage(messageText);
      textInput.value = "";
    }
  },
  render() {
    return (
      <div className="footer">
        <div className="ui form input-box">
          <div className="field">
            <textarea rows="1" ref="textInput" onKeyPress={this.handleKeyPress}></textarea>
          </div>
        </div>
      </div>
    );
  }
});
